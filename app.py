# ----------------------------------------------------------------------------
# Import ไลบรารีที่จำเป็น
# ----------------------------------------------------------------------------
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import traceback # สำหรับแสดงรายละเอียดข้อผิดพลาด (Traceback)
import numpy as np # เพิ่มเข้ามาเผื่อใช้

# ----------------------------------------------------------------------------
# การตั้งค่าและโหลดโมเดล
# ----------------------------------------------------------------------------

# --- !!! สำคัญ !!! ---
# กำหนดชื่อฟีเจอร์ (features) ที่ถูกต้อง ตามลำดับจาก DataFrame ที่ใช้เทรนโมเดล
# *** อัปเดตตาม list คอลัมน์ที่คุณให้มา (โดยเอา 'churn' ออก) ***
EXPECTED_FEATURE_NAMES = [
    'total_eve_minutes',
    'total_day_minutes',
    # 'churn', # <<--- เอาออก เพราะเป็นตัวแปรเป้าหมาย (y)
    'total_night_minutes',
    'total_intl_minutes',
    'total_night_calls',
    'customer_service_calls',
    'total_day_calls',
    'total_eve_calls',
    'total_intl_calls',
    'account_length',
    'age',
    'number_vmail_messages'
]

MODEL_FILE_PATH = 'rf_model.pkl' # ชื่อไฟล์โมเดล
model = None # กำหนดค่าเริ่มต้นให้ตัวแปร model

# --- โหลดโมเดลที่เทรนไว้แล้ว ---
try:
    model = joblib.load(MODEL_FILE_PATH)
    print(f"โหลดโมเดล '{MODEL_FILE_PATH}' สำเร็จ")
    # ตรวจสอบจำนวนฟีเจอร์ที่โมเดลคาดหวัง
    if hasattr(model, 'n_features_in_'):
        print(f"โมเดลคาดหวัง {model.n_features_in_} features.")
        if model.n_features_in_ != len(EXPECTED_FEATURE_NAMES):
            print(f"คำเตือน: จำนวนฟีเจอร์ใน EXPECTED_FEATURE_NAMES ({len(EXPECTED_FEATURE_NAMES)}) ไม่ตรงกับที่โมเดลคาดหวัง ({model.n_features_in_})!")
    # ตรวจสอบชื่อฟีเจอร์ถ้ามี (อาจไม่มีสำหรับ RandomForest แต่ใส่เผื่อ)
    if hasattr(model, 'feature_names_in_') and list(model.feature_names_in_) != EXPECTED_FEATURE_NAMES:
         print("คำเตือน: ลำดับ/ชื่อฟีเจอร์ใน EXPECTED_FEATURE_NAMES อาจไม่ตรงกับที่โมเดลถูกเทรนมา بالضبط")
         print(f"  โมเดลคาดหวัง: {list(model.feature_names_in_)}")
         print(f"  กำหนดไว้:    {EXPECTED_FEATURE_NAMES}")

except FileNotFoundError:
    print(f"เกิดข้อผิดพลาด: ไม่พบไฟล์โมเดล '{MODEL_FILE_PATH}'")
    print("โปรดตรวจสอบว่าไฟล์โมเดลอยู่ในโฟลเดอร์เดียวกับ app.py")
    model = None
except Exception as e:
    print(f"เกิดข้อผิดพลาด: ไม่สามารถโหลดโมเดล '{MODEL_FILE_PATH}' ได้")
    print(f"รายละเอียดข้อผิดพลาด: {e}")
    print("Traceback:")
    print(traceback.format_exc())
    model = None

# ----------------------------------------------------------------------------
# ตั้งค่า Flask Application
# ----------------------------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------------------------
# กำหนดเส้นทาง (Routes) ของ API
# ----------------------------------------------------------------------------

@app.route('/')
def home():
    """แสดงข้อความต้อนรับง่ายๆ สำหรับ endpoint หลัก"""
    if model is None:
        return 'Flask API กำลังทำงาน แต่ไม่สามารถโหลดโมเดล ML ได้ โปรดตรวจสอบ log ของเซิร์ฟเวอร์', 500
    else:
        feature_count = getattr(model, 'n_features_in_', len(EXPECTED_FEATURE_NAMES))
        return f'Flask API กำลังทำงานและโหลดโมเดล ML เรียบร้อยแล้ว! คาดหวัง {feature_count} features.'

@app.route('/predict', methods=['POST'])
def predict():
    """
    รับข้อมูล input ในรูปแบบ JSON, ทำนายผลโดยใช้โมเดลที่โหลดไว้,
    และคืนค่าผลการทำนายในรูปแบบ JSON.
    """
    # --- 1. ตรวจสอบว่าโมเดลถูกโหลดสำเร็จหรือไม่ ---
    if model is None:
        print("เกิดข้อผิดพลาด - /predict: โมเดลยังไม่ได้ถูกโหลด")
        return jsonify({"error": "โมเดลไม่พร้อมใช้งาน โปรดตรวจสอบ log ของเซิร์ฟเวอร์"}), 500

    # --- 2. รับและตรวจสอบข้อมูล input แบบ JSON ---
    try:
        input_data = request.get_json()
        if not input_data:
            print("เกิดข้อผิดพลาด - /predict: ไม่ได้รับข้อมูล JSON input")
            return jsonify({"error": "Request body ต้องมีข้อมูล JSON"}), 400
        print(f"ข้อมูล - /predict: ได้รับข้อมูล input: {input_data}")

    except Exception as e:
        print(f"เกิดข้อผิดพลาด - /predict: ไม่สามารถรับหรือแปลงข้อมูล JSON ได้")
        print(f"รายละเอียดข้อผิดพลาด: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({"error": "รูปแบบ JSON ใน request body ไม่ถูกต้อง"}), 400

    # --- 3. เตรียมข้อมูลสำหรับโมเดล ---
    try:
        # สร้าง DataFrame จาก dictionary ของ input
        df_input = pd.DataFrame([input_data])
        print(f"ข้อมูล - /predict: สร้าง DataFrame แล้ว มีคอลัมน์: {list(df_input.columns)}")

        # --- *** ขั้นตอนสำคัญ: ตรวจสอบและเรียงลำดับคอลัมน์ให้ตรงกับที่โมเดลคาดหวัง *** ---
        df_ordered = df_input[EXPECTED_FEATURE_NAMES]
        print(f"ข้อมูล - /predict: เรียงลำดับ/เลือกคอลัมน์ DataFrame สำหรับโมเดลแล้ว: {list(df_ordered.columns)}")
        print(f"ข้อมูล - /predict: จำนวนคอลัมน์ที่จะส่งเข้า predict: {len(df_ordered.columns)}")

    except KeyError as e:
        missing_key = str(e).strip("'")
        print(f"เกิดข้อผิดพลาด - /predict: ฟีเจอร์ที่ต้องการหายไปจาก JSON input: {missing_key}")
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({
            "error": f"ข้อมูล input ขาดฟีเจอร์ที่จำเป็น: '{missing_key}'",
            "message": "โปรดตรวจสอบว่า JSON payload มี key ครบถ้วนตามนี้",
            "required_features": EXPECTED_FEATURE_NAMES
        }), 400
    except Exception as e:
        print(f"เกิดข้อผิดพลาด - /predict: ไม่สามารถเตรียมข้อมูลสำหรับโมเดลได้")
        print(f"รายละเอียดข้อผิดพลาด: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({"error": "ไม่สามารถประมวลผลข้อมูล input ได้"}), 500

    # --- 4. ทำนายผล ---
    try:
        # ตรวจสอบจำนวน features อีกครั้งก่อน predict (เพื่อความแน่ใจ)
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(df_ordered.columns):
             print(f"เกิดข้อผิดพลาด - /predict: จำนวน features ไม่ตรง! โมเดลคาดหวัง {model.n_features_in_} แต่ได้รับ {len(df_ordered.columns)}")
             return jsonify({
                 "error": "Internal error: Feature count mismatch before prediction.",
                 "expected_count": model.n_features_in_,
                 "received_count": len(df_ordered.columns)
                 }), 500

        # ทำนายผล
        prediction_array = model.predict(df_ordered)
        prediction_result = int(prediction_array[0])
        print(f"ข้อมูล - /predict: ทำนายผลสำเร็จ ผลลัพธ์: {prediction_result}")

        # คืนค่าผลการทำนาย
        return jsonify({'prediction': prediction_result})

    except ValueError as e:
         print(f"เกิดข้อผิดพลาด - /predict: เกิด ValueError ระหว่างการทำนาย!")
         print(f"รายละเอียดข้อผิดพลาด: {e}")
         print("Traceback:")
         print(traceback.format_exc())
         # ตรวจสอบข้อความ error เพื่อให้ข้อมูลที่ชัดเจนขึ้น
         if "feature names mismatch" in str(e) or "order" in str(e):
             return jsonify({
                 "error": "เกิดข้อผิดพลาดในการทำนาย: Feature mismatch (ชื่อหรือลำดับไม่ตรง)",
                 "details": "โปรดตรวจสอบว่า EXPECTED_FEATURE_NAMES ใน app.py ตรงกับข้อมูลที่ใช้เทรนโมเดล 100% และ JSON Input ถูกต้อง",
                 "required_features_order": EXPECTED_FEATURE_NAMES,
                 "received_features_order": list(df_ordered.columns) # แสดงลำดับที่ส่งเข้า predict จริง
                 }), 400 # ควรเป็น 400 ถ้าปัญหายังเกิดจาก input/config
         else:
             # อาจเป็นปัญหา data type หรือค่าที่ไม่ถูกต้อง
             return jsonify({"error": f"การทำนายผลล้มเหลว: ค่าข้อมูลไม่ถูกต้องหรือเข้ากันไม่ได้ - {e}"}), 400
    except Exception as e:
        print(f"เกิดข้อผิดพลาด - /predict: เกิดข้อผิดพลาดที่ไม่คาดคิด")
        print(f"รายละเอียดข้อผิดพลาด: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({"error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500

# ----------------------------------------------------------------------------
# ส่วนหลักของการทำงาน (เมื่อรันไฟล์โดยตรง)
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print("กำลังเริ่ม Flask development server...")
    # **สำคัญ: ตั้งค่า debug=False สำหรับการใช้งานจริง (Production)!**
    app.run(host='0.0.0.0', port=5000, debug=True) # หรือเปลี่ยน port ถ้า 5000 ไม่ว่าง
