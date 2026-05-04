import os
import sys
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from sb3_contrib import MaskablePPO
import math
import time

from env_nn import BinPacking3DEnv

app = Flask(__name__)

CONFIG = {
    "MAX_ITEMS": 251,
    "MAX_WEIGHT": 500,
    "GRID_SIZE": 64, 
    "BIN_DIMS": (100, 70, 70),
    "SUP_THRES": 0.75,
    "MODEL_PATH": "website/models/LCVRP_env_nn_64.zip",
    "DEPOT_COORDS": (0.0, 0.0)
}

def ensure_np(obs):
    clean = {}
    for k, v in obs.items():
        if k == "action_mask": continue
        clean[k] = np.array(v, dtype=np.float32)[np.newaxis, ...]
    return clean

def ensure_mask(mask):
    return np.asanyarray(mask, dtype=np.uint8).flatten()

def policy_predict(model, obs_dict, action_mask):
    batched_mask = np.expand_dims(action_mask, axis=0).astype(bool)
    action, _ = model.predict(
        obs_dict, 
        action_masks=batched_mask, 
        deterministic=True
    )
    return int(action[0])

try:
    model = MaskablePPO.load(CONFIG["MODEL_PATH"])
    print("Model MPPO-GNN Loaded Successfully")
except Exception as e:
    print(f"Error Loading Model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proses_data', methods=['POST'])
def proses_data():
    start_time = time.time()
    temp_path = "temp_manifest.csv"
    try:
        file = request.files['file']
        file.save(temp_path)

        raw_depot = request.form.get('depot_coords', '0,0')
        try:
            depot_x, depot_y = map(float, raw_depot.split(','))
            depot_coords = (depot_x, depot_y)
        except:
            depot_coords = (0.0, 0.0)

        raw_l = request.form.get('bin_l', str(CONFIG["BIN_DIMS"][0]))
        raw_w = request.form.get('bin_w', str(CONFIG["BIN_DIMS"][1]))
        raw_h = request.form.get('bin_h', str(CONFIG["BIN_DIMS"][2]))

        m_weight = request.form.get('weight', int(CONFIG["MAX_WEIGHT"]))

        if raw_l.startswith('0') or raw_w.startswith('0') or raw_h.startswith('0'):
            return jsonify({"status": "error", "message": "Input dimensi tidak boleh diawali angka nol!"}), 400

        bin_l = float(raw_l)
        bin_w = float(raw_w)
        bin_h = float(raw_h)
        
        if bin_l <= 0 or bin_w <= 0 or bin_h <= 0:
            return jsonify({"status": "error", "message": "Dimensi kontainer harus positif!"}), 400

        df = pd.read_csv(temp_path, keep_default_na=False)
        
        if 'id' not in df.columns:
            return jsonify({"status": "error", "message": "Kolom id tidak ada di CSV"}), 400
        
        total_actual_items = df['quantity'].sum()
        if total_actual_items > CONFIG["MAX_ITEMS"]:
            return jsonify({
                "status": "error", 
                "message": f"Total barang ({total_actual_items}) melebihi batas {CONFIG['MAX_ITEMS']}."
            }), 400
        
        id_series = df['id'].astype(str).str.strip()

        if (id_series == '').any() or (id_series.str.lower() == 'nan').any():
            return jsonify({
                "status": "error", 
                "message": "Kolom id tidak boleh kosong, hanya spasi, atau kutip kosong"
            }), 400
        
        if not id_series.str.isalnum().all():
            return jsonify({
                "status": "error", 
                "message": "Kolom id hanya boleh berisi huruf atau angka (tidak boleh simbol/spasi!)"
            }), 400
        
        numeric_cols = ['quantity', 'length', 'width', 'height', 'weight', 'fragile', 'dest_x', 'dest_y']
        
        non_negative_cols = ['quantity', 'length', 'width', 'height', 'weight', 'fragile']
        
        for col in numeric_cols:
            if col not in df.columns:
                return jsonify({"status": "error", "message": f"Kolom {col} tidak ada di CSV"}), 400
            
            converted_col = pd.to_numeric(df[col], errors='coerce')
            if converted_col.isna().any():
                return jsonify({
                    "status": "error", 
                    "message": f"Kolom {col} mengandung simbol atau karakter non-angka"
                }), 400
            
            if col in non_negative_cols:
                if (converted_col < 0).any():
                    return jsonify({
                        "status": "error", 
                        "message": f"Data di kolom {col} tidak boleh bernilai negatif"
                    }), 400

        try:
            m_weight = float(m_weight)
            if m_weight <= 0:
                return jsonify({"status": "error", "message": "Kapasitas berat harus lebih dari 0!"}), 400
        except ValueError:
            m_weight = float(CONFIG["MAX_WEIGHT"])
                
            max_item_l = df['length'].max()
            max_item_w = df['width'].max()
            max_item_h = df['height'].max()

            if max_item_l > bin_l or max_item_w > bin_w or max_item_h > bin_h:
                too_big = df[(df['length'] > bin_l) | (df['width'] > bin_w) | (df['height'] > bin_h)].iloc[0]
                msg = (f"Bin terlalu kecil! Barang ID '{too_big['id']}' ({int(too_big['length'])}x{int(too_big['width'])}x{int(too_big['height'])}) "
                    f"melebihi dimensi kontainer ({int(bin_l)}x{int(bin_w)}x{int(bin_h)}).")
                return jsonify({"status": "error", "message": msg}), 400

        current_bin_dims = (bin_l, bin_w, bin_h)
        sup_thres = float(request.form.get('support_threshold', 75)) / 100.0 

        eval_env = BinPacking3DEnv(
            csv_path=temp_path,
            bin_dims=current_bin_dims,
            max_weight=m_weight,
            grid_size=CONFIG["GRID_SIZE"],
            support_threshold=sup_thres,
            max_items=CONFIG["MAX_ITEMS"],
            depot_coords=depot_coords
        )

        _, _ = eval_env.reset()
        global_availability_mask = eval_env.items_state[:, 0].copy()
        
        all_bins_data = [] 
        bin_count = 0
        total_distance = 0

        while np.sum(global_availability_mask) > 0.5:
            bin_count += 1
            obs, _ = eval_env.reset()
            eval_env.items_state[:, 0] = global_availability_mask.copy()
            
            for i in range(eval_env.max_items):
                if eval_env.items_state[i, 0] == 0.0:
                    eval_env.items_state[i, 4] = -1.0 

            terminated = False
            while not terminated:
                obs = eval_env._get_obs()
                if np.sum(obs['action_mask']) == 0:
                    break
                    
                action_mask = ensure_mask(obs["action_mask"])
                action = policy_predict(model, ensure_np(obs), action_mask)
                obs, reward, terminated, truncated, info = eval_env.step(action)

            final_packed_items = eval_env.finalize_bin() 
            global_availability_mask = eval_env.items_state[:, 0].copy()

            if not final_packed_items:
                break

            items_in_this_bin = []
            bin_weight = 0
            total_item_volume = 0
            bin_volume = bin_l * bin_w * bin_h

            scale_l, scale_w, scale_h = bin_l/CONFIG["GRID_SIZE"], bin_w/CONFIG["GRID_SIZE"], bin_h/CONFIG["GRID_SIZE"]

            for it in final_packed_items:
                orig_l = int(it.get('length', it['dims'][0]))
                orig_w = int(it.get('width', it['dims'][1]))
                orig_h = int(it.get('height', it['dims'][2]))
                total_item_volume += (orig_l * orig_w * orig_h)
                
                visual_l = (math.ceil(it['dims'][0] / scale_l) * scale_l) - 0.1
                visual_w = (math.ceil(it['dims'][1] / scale_w) * scale_w) - 0.1
                visual_h = (math.ceil(it['dims'][2] / scale_h) * scale_h) - 0.1

                w_kg = int(it.get('weight', 0))
                bin_weight += w_kg

                items_in_this_bin.append({
                    "id": str(it.get('id', 'N/A')).split('.')[0],
                    "l_orig": orig_l, "w_orig": orig_w, "h_orig": orig_h,
                    "l": visual_l, "w": visual_w, "h": visual_h,
                    "x": int(it.get('dest_x') or 0), "y": int(it.get('dest_y') or 0),
                    "weight": w_kg,
                    "pos3d": [
                        int(it['pos'][1]) * scale_w, 
                        int(it['pos'][2]) * scale_h, 
                        int(it['pos'][0]) * scale_l  
                    ],
                    "bin_id": bin_count
                })

            bin_distance = 0
            if items_in_this_bin:
                try:
                    dx, dy = depot_coords 
                    
                    coords = [(float(i['x']), float(i['y'])) for i in items_in_this_bin]
                    bin_distance += math.sqrt((coords[0][0] - dx)**2 + (coords[0][1] - dy)**2)
                    for i in range(len(coords) - 1):
                        bin_distance += math.sqrt((coords[i+1][0] - coords[i][0])**2 + (coords[i+1][1] - coords[i][1])**2)
                    bin_distance += math.sqrt((coords[-1][0] - dx)**2 + (coords[-1][1] - dy)**2)
                    total_distance += bin_distance

                except Exception as e:
                    print(f"Error hitung jarak: {e}")
                    bin_distance = 0

            volume_eff = (total_item_volume / bin_volume) * 100 if bin_volume > 0 else 0
            all_bins_data.append({
                "bin_id": bin_count,
                "items": items_in_this_bin,
                "stats": {
                    "volume_eff": f"{round(volume_eff, 2)}",
                    "total_weight": f"{round(bin_weight, 2)}",
                    "count": len(items_in_this_bin),
                    "distance": f"{round(bin_distance, 2)}",
                }
            })

        return jsonify({
            "status": "success",
            "bins": all_bins_data,
            "bin_dims": [bin_l, bin_w, bin_h],
            "execution_time": int((time.time() - start_time) * 1000),
            "total_distance": f"{round(total_distance, 2)}"
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)