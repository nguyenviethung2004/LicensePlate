from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import io
import numpy as np
from PIL import Image
import face_recognition
from supabase import create_client, Client
import os
from ultralytics import YOLO
from LicensePlateDetect.predict import predict_and_crop
from LPRNetOCR.license_plate_reader import process_license_plate
import os
from dotenv import load_dotenv
import tempfile
from datetime import datetime
import time
import ast


app = Flask(__name__)
CORS(app)  


base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "model_dir")

model_detect = YOLO(os.path.join(model_dir, "DetectLicensePlate.pt"))
model_ocr = YOLO(os.path.join(model_dir, "OrcLicensePlate.pt"))


app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def read_image_file(file_storage) -> np.ndarray:

    image = Image.open(io.BytesIO(file_storage.read()))
    image = image.convert("RGB") 
    return np.array(image)

def predict_license_plate(model_detect, model_ocr, img_array):
    try:
        crop_img = predict_and_crop(model_detect, img_array, conf=0.25)
        if crop_img is None:
            return None, "Không phát hiện được biển số."

        results = model_ocr.predict(source=crop_img, conf=0.5, verbose=False)
        license_plate, len_rows = process_license_plate(results)
        if license_plate is None:
            return None, "Không đọc được biển số."

        return license_plate
    except Exception as e:
        return None, f"Lỗi khi xử lý biển số: {str(e)}"

def extract_face_vector(image: np.ndarray) -> str:

    try:
        face_encodings = face_recognition.face_encodings(image, num_jitters=1, model='small')
        
        if not face_encodings:
            raise ValueError("Không thể trích xuất đặc trưng khuôn mặt")

        face_vector = face_encodings[0].tolist()
        return str(face_vector)
    
    except Exception as e:
        print(f"Error extracting face vector: {e}")
        return str([0.0] * 128)



def save_to_database(plate_number: str, face_vector: str, record_type: str, mode: str = "insert", record_id: int = None) -> dict:

    try:
        current_time = datetime.utcnow().isoformat()
        
        data = {
            "plate_number": plate_number,
            "face_vector": face_vector,
            "type": record_type,
        }

        if record_type == "check-in":
            data["check_in_time"] = current_time
        else:
            data["check_out_time"] = current_time

        if mode == "insert":
            response = supabase.table("records").insert(data).execute()
        elif mode == "update":
            if not record_id:
                raise ValueError("Thiếu record_id khi update")
            response = supabase.table("records").update(data).eq("id", record_id).execute()
        else:
            raise ValueError("mode phải là 'insert' hoặc 'update'")

        if response.data:
            return response.data[0]
        else:
            raise Exception("Không thể lưu vào database")

    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "id": record_id or 9999,
            "plate_number": plate_number,
            "face_vector": face_vector,
            "type": record_type,
            "check_in_time": current_time if record_type == "check-in" else None,
            "check_out_time": current_time if record_type == "check-out" else None,
        }



@app.route('/api/checkin', methods=['POST'])
def checkin():
    try:
        start_time = time.time()
        if 'licensePlateImage' not in request.files:
            
            return jsonify({"error": "Thiếu ảnh biển số xe"}), 400
        if 'faceImage' not in request.files:
            return jsonify({"error": "Thiếu ảnh khuôn mặt"}), 400
        
        plate_file = request.files['licensePlateImage']
        face_file = request.files['faceImage']
        
        print(request.files)

        if plate_file.filename == '' or face_file.filename == '':
            return jsonify({"error": "File rỗng"}), 400
        
        
        face_img = read_image_file(face_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            plate_file.save(tmp.name)
            plate_file_1 = tmp.name

        
        plate_number = predict_license_plate(model_detect, model_ocr, plate_file_1)
        elapsed_time = time.time() - start_time

        start_time2 = time.time()
        face_vector = extract_face_vector(face_img)
        elapsed_time2 = time.time() - start_time2
        print(f"Time to extract face vector: {elapsed_time2:.2f} seconds")

        if plate_number is None or len(plate_number.strip()) == 0:
            return jsonify({"error": "Không nhận dạng được biển số"}), 400
        if face_vector is None or len(face_vector) == 0:
            return jsonify({"error": "Không phát hiện được khuôn mặt"}), 400
        
        start_time1 = time.time()
        data = supabase.table("records")\
            .select("*")\
            .eq("plate_number", plate_number)\
            .eq("type", "check-in")\
            .is_("check_out_time", "null")\
            .order("check_in_time", desc=True)\
            .limit(1)\
            .execute()
        
        if data.data:
            return jsonify({"error": f"Tìm thấy xe có biển số {plate_number} đang đỗ đã check-in"}), 404
        
        result = save_to_database(plate_number, face_vector, "check-in", "insert")
        elapsed_time1 = time.time() - start_time1
        return jsonify({
            "success": True,
            "message": "Check-in thành công",
            "time":elapsed_time+elapsed_time1+elapsed_time2,
            **result
        }), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500



@app.route('/api/checkout', methods=['POST'])
def checkout():
    try:
        start_time = time.time()
        if 'licensePlateImage' not in request.files:
            return jsonify({"error": "Thiếu ảnh biển số xe"}), 400
        if 'faceImage' not in request.files:
            return jsonify({"error": "Thiếu ảnh khuôn mặt"}), 400
        
        plate_file = request.files['licensePlateImage']
        face_file = request.files['faceImage']


        if plate_file.filename == '' or face_file.filename == '':
            return jsonify({"error": "File rỗng"}), 400
        
        face_img = read_image_file(face_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            plate_file.save(tmp.name)
            plate_file_1 = tmp.name

        plate_number = predict_license_plate(model_detect, model_ocr, plate_file_1)
        face_vector = extract_face_vector(face_img)


        if plate_number is None or len(plate_number.strip()) == 0:
            return jsonify({"error": "Không nhận dạng được biển số"}), 400
        if face_vector is None or len(face_vector) == 0:
            return jsonify({"error": "Không phát hiện được khuôn mặt"}), 400
        
        data = supabase.table("records")\
            .select("*")\
            .eq("plate_number", plate_number)\
            .eq("type", "check-in")\
            .is_("check_out_time", "null")\
            .order("check_in_time", desc=True)\
            .limit(1)\
            .execute()
        
        if not data.data:
            return jsonify({"error": f"Không tìm thấy xe có biển số {plate_number} đang đỗ"}), 404
        
        record = data.data[0]
        face_stored_vector= record["face_vector"]

        saved_face_vector = np.array(ast.literal_eval(face_stored_vector))
        current_face_vector = np.array(ast.literal_eval(face_vector))

        match_face  = face_recognition.compare_faces([saved_face_vector], current_face_vector, tolerance=0.5)
        distance = face_recognition.face_distance([saved_face_vector], current_face_vector)
        
        if not match_face[0]:
            return jsonify({
                "success": False,
                "error": "Khuôn mặt không khớp với lần check-in",
                "distance": float(distance)
            }), 403
        
        result = save_to_database(plate_number, face_vector, "check-out", "update", record["id"])
        elapsed_time = time.time() - start_time
        return jsonify({
            "success": True,
            "message": "Check-out thành công",
            "distance": float(distance),
            "time":elapsed_time,
            **result
        }), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500



if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )