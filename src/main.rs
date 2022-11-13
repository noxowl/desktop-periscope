use std::path::Path;
use anyhow::{Result, Context, Error};
use opencv::{self as cv, prelude::*, videoio, imgproc};
use opencv::core::{Point, Rect, Scalar, Size, Vector, VectorExtern};
use opencv::videoio::VideoCapture;

fn main() {
    let mut cam: VideoCapture = VideoCapture::default().unwrap();
    let mut frame: Mat = Default::default();
    // find_cameras().unwrap();
    let cam_type = videoio::CAP_ANY;
    if cam_type == videoio::CAP_OPENNI2 {
        setup_camera(&mut cam, 3, videoio::CAP_OPENNI2).unwrap();
    } else {
        setup_camera(&mut cam, 1, videoio::CAP_ANY).unwrap();
    }
    loop {
        match cam.is_opened() {
            Ok(_) => {
                capture_from_camera(&mut cam, &mut frame).unwrap();
                analyse_and_process_from_cv(&mut frame);
                cv::highgui::imshow("Periscope", &frame).unwrap();
            }
            Err(_) => { panic!("open capture device failed!") }
        }
        let key = cv::highgui::wait_key(60).unwrap();
        if key > 0 && key != 255 {
            break;
        }
    }
}

fn find_cameras() -> Result<Vec<i32>, Error> {
    // TODO: handle SIGSEGV
    let mut list_of_cameras: Vec<i32> = vec!();
    let mut cam = VideoCapture::default().unwrap();
    let mut current = 0;
    loop {
        let opened = cam.open(*&current, videoio::CAP_ANY);
        match opened {
            Ok(o) => {
                if o {
                    cam.release().context("release camera failed!").unwrap();
                    list_of_cameras.push(*&current);
                } else {
                    break
                }
            }
            Err(_) => {
                break
            }
        }
        current += 1;
    }
    println!("{:?}", &list_of_cameras);
    Ok(list_of_cameras)
}

fn setup_camera(cam: &mut VideoCapture, camera_index: i32, camera_type: i32) -> Result<bool, Error> {
    *cam = VideoCapture::new(camera_index, camera_type).unwrap();
    if camera_type == videoio::CAP_ANY {
        cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.).unwrap();
        cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.).unwrap();
    }
    Ok(true)
}

fn capture_from_camera(cam: &mut VideoCapture, mut frame: &mut Mat) -> Result<bool, Error> {
    match cam.is_opened() {
        Ok(_) => {
            let mut frame_dst: Mat = Default::default();
            cam.grab().unwrap();
            match cam.get_backend_name()?.as_str() {
                "OPENNI2" => {
                    cam.retrieve(&mut frame, videoio::CAP_OPENNI_BGR_IMAGE).unwrap();
                    println!("{}:{}", frame.cols(), frame.rows());
                    imgproc::resize(
                        frame,
                        &mut frame_dst,
                        Size::new(854, 480),
                        0., // fx. default 0.
                        0., // fy. default 0.
                        imgproc::INTER_LINEAR as i32
                    ).unwrap();
                    frame_dst.copy_to(*&mut frame).unwrap();
                    cv::core::flip(frame, &mut frame_dst, 1).unwrap();
                    frame_dst.copy_to(*&mut frame).unwrap();
                }
                _ => {
                    cam.retrieve(&mut frame, videoio::CAP_DSHOW).unwrap();
                    let mut frame_dst: Mat = Default::default();
                    cv::core::flip(frame, &mut frame_dst, 1).unwrap();
                    frame_dst.copy_to(*&mut frame).unwrap();
                }
            }
            Ok(true)
        }
        Err(e) => { Err(anyhow::Error::from(e)) }
    }
}

fn find_center_of_frame(frame: &mut Mat) -> Result<Point, Error> {
    Ok(Point::new(frame.cols() / 2,frame.rows() / 2))
}

fn analyse_and_process_from_cv(mut frame: &mut Mat) {
    let center = find_center_of_frame(frame).unwrap();
    let faces = find_face(frame).unwrap();
    let mut number_of_faces = 0;
    unsafe {
        number_of_faces = faces.extern_len();
    };
    println!("number of detected faces: {:?}", number_of_faces);
    if number_of_faces > 0 {
        let mut rest_of_faces: Vec<Rect> = Default::default();
        let nearest_face = pop_nearest_face(&faces, &mut rest_of_faces, &center).unwrap();
        println!("number of rest of faces: {}", &rest_of_faces.len());
        let mut nearest_face_roi = Mat::roi(
            &frame,
            *&nearest_face).unwrap();
        let mut nearest_face_roi_dst: Mat = Default::default();
        blur_nearest_face(&nearest_face_roi, &mut nearest_face_roi_dst);
        nearest_face_roi_dst.copy_to(&mut nearest_face_roi).unwrap();
        imgproc::rectangle(
            &mut frame,
            nearest_face,
            Scalar::new(0.0, 255.0, 255.0, 100.0),
            2, // thickness
            imgproc::LineTypes::LINE_8 as i32,
            0 // shift. number of fractional bits in the point coordinates
        ).unwrap();
        for face in rest_of_faces {
            imgproc::rectangle(
                &mut frame,
                face,
                Scalar::new(0.0, 255.0, 0.0, 100.0),
                2, // thickness
                imgproc::LineTypes::LINE_8 as i32,
                0 // shift. number of fractional bits in the point coordinates
            ).unwrap();
        }
    }
}

fn find_face(frame: &Mat) -> Result<Vector<Rect>> {
    let opencv_path = "/usr/local/share/opencv4/";
    let mut face_cascade = cv::objdetect::CascadeClassifier::new(
        Path::new(opencv_path).join("haarcascades/haarcascade_frontalface_default.xml").to_str().unwrap())?;
    let mut face_cascade_result: Vector<Rect> = Default::default();
    face_cascade.detect_multi_scale(
        frame,
        &mut face_cascade_result,
        1.3, // scale_factor
        5, // min_neighbors
        opencv::objdetect::CASCADE_SCALE_IMAGE, // flags
        Default::default(),// min_size
        Default::default()// max_size
    ).unwrap();
    Ok(face_cascade_result)
}

fn pop_nearest_face(faces: &Vector<Rect>, rest_of_faces: &mut Vec<Rect>, center: &Point) -> Result<Rect> {
    let mut nearest_rect: Rect = Default::default();
    let mut nearest_diff: Point = Default::default();
    let mut number_of_faces = 0;
    unsafe {
        number_of_faces = faces.extern_len();
    };
    for rect in faces {
        if rect.x > 0 && rect.y > 0 {
            let center_of_rect = Point::from((rect.br() + rect.tl()) / 2_i32);
            let diff = Point::new(
                center_of_rect.x.abs_diff(center.x) as i32,
                center_of_rect.y.abs_diff(center.y) as i32
            );
            if diff.x == 0 && diff.y == 0 && number_of_faces == 1 {
                return Ok(rect)
            } else if nearest_diff.x == 0 && nearest_diff.y == 0 {
                nearest_diff = diff;
                nearest_rect = rect;
            } else if nearest_diff > diff {
                rest_of_faces.push(nearest_rect);
                nearest_diff = diff;
                nearest_rect = rect;
            } else {
                rest_of_faces.push(rect);
            }
        }
    }
    Ok(nearest_rect)
}

fn blur_nearest_face(face: &Mat, mut face_dst: &mut Mat) {
    imgproc::gaussian_blur(
        &face,
        &mut face_dst,
        Size::new(55, 55),
        10000.,
        0., // the default value of sigma Y is 0.
        cv::core::BORDER_DEFAULT
    ).unwrap();
}
