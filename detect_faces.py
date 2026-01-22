import cv2
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not installed. Install it with: pip install mediapipe")
    exit(1)


class FaceDetectorAndCropper:
    """
    Detects faces in images using MediaPipe and crops them for dataset preparation.
    Suitable for diverse face datasets including Indian faces.
    """
    
    def __init__(self, output_dir: str = "cropped_faces", min_face_size: int = 50):
        """
        Initialize the face detector and cropper.
        
        Args:
            output_dir: Directory to save cropped face images
            min_face_size: Minimum face size (width/height) to consider valid detection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_face_size = min_face_size
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short-range, 1=full-range (better for varied distances)
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_and_crop_faces(
        self, 
        image_path: str, 
        padding: float = 0.2,
        save_annotated: bool = False
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Detect faces in an image and crop them.
        
        Args:
            image_path: Path to the input image
            padding: Padding around detected face (0.2 = 20% extra space)
            save_annotated: If True, save image with face boxes drawn
            
        Returns:
            cropped_faces: List of cropped face images (numpy arrays)
            face_info: List of dictionaries with face detection info
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return [], []
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        
        # Detect faces
        results = self.face_detection.process(image_rgb)
        
        cropped_faces = []
        face_info = []
        
        if results.detections:
            for idx, detection in enumerate(results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                box_width = int(bbox.width * w)
                box_height = int(bbox.height * h)
                
                # Check minimum face size
                if box_width < self.min_face_size or box_height < self.min_face_size:
                    print(f"Face {idx} too small ({box_width}x{box_height}). Skipping.")
                    continue
                
                # Add padding
                pad_x = int(box_width * padding)
                pad_y = int(box_height * padding)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + box_width + pad_x)
                y2 = min(h, y + box_height + pad_y)
                
                # Crop face
                cropped_face = image[y1:y2, x1:x2]
                cropped_faces.append(cropped_face)
                
                # Store face info
                face_data = {
                    'face_id': idx,
                    'confidence': float(detection.score[0]),
                    'bbox_pixel': {
                        'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1
                    },
                    'bbox_normalized': {
                        'xmin': bbox.xmin,
                        'ymin': bbox.ymin,
                        'width': bbox.width,
                        'height': bbox.height
                    }
                }
                face_info.append(face_data)
        
        # Save annotated image if requested
        if save_annotated and results.detections:
            annotated_image = image.copy()
            for detection in results.detections:
                self.mp_drawing.draw_detection(annotated_image, detection)
            
            image_name = Path(image_path).stem
            annotated_path = self.output_dir / f"{image_name}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_image)
        
        return cropped_faces, face_info
    
    def process_dataset(
        self,
        input_dir: str,
        metadata_file: str = None,
        padding: float = 0.2,
        save_annotated: bool = False
    ) -> Dict:
        """
        Process all images in a directory, detect and crop faces.
        
        Args:
            input_dir: Directory containing input images
            metadata_file: Optional JSON file to save detection metadata
            padding: Padding around detected faces
            save_annotated: If True, save annotated images
            
        Returns:
            results: Dictionary with processing statistics and metadata
        """
        input_path = Path(input_dir)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"Found {len(image_files)} images. Processing...")
        
        results = {
            'total_images': len(image_files),
            'total_faces_detected': 0,
            'images_with_faces': 0,
            'images_without_faces': 0,
            'per_image': {}
        }
        
        metadata = {}
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing {image_file.name}...")
            
            cropped_faces, face_info = self.detect_and_crop_faces(
                str(image_file),
                padding=padding,
                save_annotated=save_annotated
            )
            
            if cropped_faces:
                results['images_with_faces'] += 1
                results['total_faces_detected'] += len(cropped_faces)
                
                # Save cropped faces
                image_stem = image_file.stem
                image_output_dir = self.output_dir / image_stem
                image_output_dir.mkdir(exist_ok=True)
                
                for face_idx, (cropped_face, face_data) in enumerate(
                    zip(cropped_faces, face_info)
                ):
                    face_filename = f"{image_stem}_face_{face_idx:03d}.jpg"
                    face_path = image_output_dir / face_filename
                    cv2.imwrite(str(face_path), cropped_face)
                    
                    # Update metadata
                    face_data['saved_path'] = str(face_path)
                
                metadata[image_file.name] = face_info
                results['per_image'][image_file.name] = {
                    'faces_detected': len(cropped_faces),
                    'faces_info': face_info
                }
            else:
                results['images_without_faces'] += 1
                print(f"  No faces detected in {image_file.name}")
        
        # Save metadata
        if metadata_file:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"\nMetadata saved to {metadata_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images processed: {results['total_images']}")
        print(f"Images with faces: {results['images_with_faces']}")
        print(f"Images without faces: {results['images_without_faces']}")
        print(f"Total faces detected: {results['total_faces_detected']}")
        if results['total_images'] > 0:
            print(f"Average faces per image: {results['total_faces_detected']/results['total_images']:.2f}")
        print(f"Cropped faces saved to: {self.output_dir}")
        print("="*60)
        
        return results
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        self.face_detection.close()


def main():
    """Example usage of the FaceDetectorAndCropper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and crop faces from images")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cropped_faces",
        help="Directory to save cropped faces (default: cropped_faces)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="JSON file to save detection metadata"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Padding around faces as fraction of face size (default: 0.2)"
    )
    parser.add_argument(
        "--min_face_size",
        type=int,
        default=50,
        help="Minimum face size in pixels (default: 50)"
    )
    parser.add_argument(
        "--save_annotated",
        action="store_true",
        help="Save images with face boxes drawn"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FaceDetectorAndCropper(
        output_dir=args.output_dir,
        min_face_size=args.min_face_size
    )
    
    # Process dataset
    detector.process_dataset(
        input_dir=args.input_dir,
        metadata_file=args.metadata,
        padding=args.padding,
        save_annotated=args.save_annotated
    )


if __name__ == "__main__":
    main()
