import eventlet
eventlet.monkey_patch()

import os
import sys
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from sb3_contrib import MaskablePPO
import math
import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import base64
import io
from flask_socketio import SocketIO, emit

from env_nn import BinPacking3DEnv

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

CONFIG = {
    "MAX_ITEMS": 251,
    "MAX_WEIGHT": 500,
    "GRID_SIZE": 64, 
    "BIN_DIMS": (100, 70, 70),
    "SUP_THRES": 0.75,
    "MODEL_PATH": "website/models/LCVRP_env_nn_64.zip",
    "DEPOT_COORDS": (0.0, 0.0)
}

CONFIG_GRB = {
    "vehicle_capacity_weight": 100000,
    "num_vehicles": 3,
    "distance_penalty": 1000,
    "gap": 0.50
}

def run_gurobi_background(temp_path, ds_name, size_str, total_distance_drl, drl_time, depot_coord, bin_l, bin_w, bin_h):
    try:
        socketio.emit('gurobi_status', {'message': 'Gurobi sedang menghitung rute...'})
        
        if not os.path.exists(temp_path):
            return

        df_to_solve = pd.read_csv(temp_path)
        
        grb_result = solve_3l_cvrp(df_to_solve, depot_coord, bin_l, bin_w, bin_h)
        
        grb_dist_val = 0
        grb_runtime_val = 0
        grb_data = []
        grb_stats = {"total_distance": 0, "runtime": 0}

        if grb_result and grb_result.get("success"):
            grb_dist_val = grb_result["stats"].get("total_distance", 0)
            grb_runtime_val = grb_result["stats"].get("runtime", 0)
            grb_data = grb_result.get("data", [])
            grb_stats = grb_result.get("stats", grb_stats)

        comparison_data = {
            'dataset_name': [ds_name],
            'container_size': [size_str],
            'dist_gurobi': [grb_dist_val],
            'dist_drl': [total_distance_drl],
            'time_gurobi': [grb_runtime_val],
            'time_drl': [drl_time]
        }
        df_temp = pd.DataFrame(comparison_data)
        
        eventlet.sleep(0) 
        
        img_data = visualize_to_base64(df_temp)

        eventlet.sleep(0) 
        
        socketio.emit('gurobi_finished', {
            'image': img_data,
            'dataset': ds_name,
            'bins': grb_data,
            'stats': grb_stats,
            'success': True
        })

    except Exception as e:
        print(f"Error pada run_gurobi_background: {str(e)}")
        socketio.emit('gurobi_error', {'message': f"Gurobi gagal: {str(e)}"})
        
    finally:
        if os.path.exists(temp_path): 
            try:
                os.remove(temp_path)
            except:
                pass

def gap_callback(model, where):
    eventlet.sleep(0) 

    t_limit = 60

    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        current_time = model.cbGet(GRB.Callback.RUNTIME)
        
        model._actual_runtime = current_time 

        if abs(obj_best - model._last_obj) > 1e-4:
            model._last_obj = obj_best
            model._last_time = current_time

        if (current_time - model._last_time) > t_limit:
            print(f"Terminating: Tidak ada perubahan dari Best Objective selama {t_limit} detik.")
            model.terminate()

        now = time.time()
        if not hasattr(model, "_last_emit_time") or (now - model._last_emit_time > 2.0):
            try:
                gap = 100.0
                if abs(obj_best) < 1e30: 
                    if abs(obj_best) > 1e-10:
                        gap = abs(obj_bound - obj_best) / abs(obj_best) * 100
                    else:
                        gap = abs(obj_bound - obj_best) * 100
                
                socketio.emit('gurobi_progress', {
                    'obj': round(obj_best, 2) if abs(obj_best) < 1e30 else "Mencari...",
                    'gap': round(gap, 2) if abs(obj_best) < 1e30 else 100,
                    'runtime': round(current_time, 1),
                    'no_improve': round(current_time - model._last_time, 1)
                })
                model._last_emit_time = now
            except Exception as e:
                pass

def solve_3l_cvrp(temp_file, depot_pos, bin_l, bin_w, bin_h):
    # ==========================================
    # 1. LOADING
    # ==========================================
    df_raw = temp_file

    grouping_map = df_raw.groupby(['dest_x', 'dest_y'])['id'].apply(
        lambda x: ", ".join(map(str, sorted(x.unique())))
    ).to_dict()

    df_raw['combined_id'] = df_raw.apply(
        lambda r: grouping_map[(r['dest_x'], r['dest_y'])], axis=1
    )

    depot_id = -1
    customers = sorted(df_raw['combined_id'].unique().tolist())
    nodes = [depot_id] + customers
    num_nodes = len(nodes)

    coords = {depot_id: depot_pos}
    node_coords = df_raw[['combined_id', 'dest_x', 'dest_y']].drop_duplicates('combined_id')
    for _, row in node_coords.iterrows():
        coords[row['combined_id']] = (float(row['dest_x']), float(row['dest_y']))

    dist = lambda u, v: np.sqrt((coords[u][0]-coords[v][0])**2 + (coords[u][1]-coords[v][1])**2)

    items = []
    item_to_customer = {}
    item_data = {}
    
    for _, row in df_raw.iterrows():
        c_node_id = row['combined_id'] 
        original_id = int(row['id'])
        
        for q in range(int(row['quantity'])):
            item_id = f"item_{original_id}_{q}" 
            items.append(item_id)
            
            item_to_customer[item_id] = c_node_id 
            item_data[item_id] = row.to_dict()

    MW = bin_l
    MH = bin_w
    MD = bin_h

    # ==========================================
    # 2. INISIALISASI MODEL
    # ==========================================
    total_threads = os.cpu_count() or 1
    
    model = gp.Model("3L-CVRP_Optimized")
    # model.setParam('MIPGap', CONFIG_GRB["gap"])
    model.setParam('Symmetry', 2)
    # model.setParam('Method', 2)
    grb_threads = max(1, total_threads - 0)
    model.setParam('Threads', grb_threads)

    model._last_obj = GRB.INFINITY
    model._last_time = 0

    # ==========================================
    # 3. VARIABEL KEPUTUSAN
    # ==========================================
    V = nodes
    V_no_depot = customers
    B = range(CONFIG_GRB["num_vehicles"])
    I = items
    orientations = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    O = range(len(orientations))

    r = model.addVars(V, V, B, vtype=GRB.BINARY, name="r")
    p = model.addVars(V, B, vtype=GRB.BINARY, name="p")
    s = model.addVars(V, lb=0, vtype=GRB.CONTINUOUS, name="s")
    f = model.addVars(V, V, vtype=GRB.BINARY, name="f")
    b = model.addVars(B, vtype=GRB.BINARY, name="b")

    x = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="z")

    x_prime = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="x_p")
    y_prime = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="y_p")
    z_prime = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="z_p")

    o = model.addVars(I, O, vtype=GRB.BINARY, name="o")
    g = model.addVars(I, vtype=GRB.BINARY, name="g")
    l_jk = model.addVars(I, I, range(6), vtype=GRB.BINARY, name="ljk")

    # ==========================================
    # 4. OBJECTIVE FUNCTION
    # ==========================================
    
    obj = gp.quicksum(dist(u, v) * r[u, v, i] for u in V for v in V if u != v for i in B) \
          + gp.quicksum(CONFIG_GRB["distance_penalty"] * b[i] for i in B) \
          + gp.quicksum(z[j] * 0.001 for j in I)
    model.setObjective(obj, GRB.MINIMIZE)

    # ==========================================
    # 5. KENDALA
    # ==========================================
    
    for v in V_no_depot:
        model.addConstr(
            gp.quicksum(r[u, v, i] for u in V if u != v for i in B) == 1, 
            name=f"visit_once_{v}"
        )

    for i in B:
        # Konservasi Aliran: Jika kendaraan masuk ke v, ia harus keluar dari v
        for v in V:
            model.addConstr(
                gp.quicksum(r[u, v, i] for u in V if u != v) == 
                gp.quicksum(r[v, u, i] for u in V if u != v),
                name=f"flow_{v}_{i}"
            )

        # Setiap kendaraan yang digunakan MAKSIMAL keluar dari depot 1 kali
        model.addConstr(
            gp.quicksum(r[depot_id, v, i] for v in V_no_depot) <= 1,
            name=f"start_depot_{i}"
        )

        # Hubungkan variabel r (routing) dengan p (assignment kendaraan)
        # Jika ada rute ke u di kendaraan i, maka p[u, i] harus 1
        for u in V_no_depot:
            model.addConstr(
                gp.quicksum(r[u, v, i] for v in V if u != v) == p[u, i],
                name=f"link_r_p_{u}_{i}"
            )

    # MTZ Subtour Elimination (Diperketat)
    # Menentukan urutan kunjungan s[v] untuk mencegah rute melingkar yang tidak ke depot
    for i in B:
        for u in V:
            for v in V_no_depot:
                if u != v:
                    model.addConstr(
                        s[v] >= s[u] + 1 - num_nodes * (1 - r[u, v, i]),
                        name=f"mtz_{u}_{v}_{i}"
                    )

    # Variabel f[u, v] (Precedence): 1 jika u dikunjungi sebelum v di kendaraan yang sama
    for u in V_no_depot:
        for v in V_no_depot:
            if u == v: continue
            
            # f[u,v] + f[v,u] = 1 HANYA JIKA u dan v berada di kendaraan i yang sama
            model.addConstr(f[u, v] + f[v, u] >= gp.quicksum(p[u, i] + p[v, i] - 1 for i in B))
            model.addConstr(f[u, v] + f[v, u] <= 1)
            
            # Menghubungkan urutan s dengan variabel biner precedence f
            model.addConstr(s[u] - s[v] <= -1 + num_nodes * (1 - f[u, v]), name=f"precedence_{u}_{v}")

    # --- Fragility ---
    for j in I:
        f_j = item_data[j]['fragile']
        model.addConstr(gp.quicksum(l_jk[j, k, 4] for k in I if j != k) <= len(I) * (1 - f_j))

    # --- Dimensions & Orientations ---
    for j in I:
        dims = [item_data[j]['length'], item_data[j]['width'], item_data[j]['height']]
        model.addConstr(gp.quicksum(o[j, r_idx] for r_idx in O) == 1)
        model.addConstr(x_prime[j] == gp.quicksum(dims[orientations[r_idx][0]] * o[j, r_idx] for r_idx in O))
        model.addConstr(y_prime[j] == gp.quicksum(dims[orientations[r_idx][1]] * o[j, r_idx] for r_idx in O))
        model.addConstr(z_prime[j] == gp.quicksum(dims[orientations[r_idx][2]] * o[j, r_idx] for r_idx in O))
        
        model.addConstr(x[j] + x_prime[j] <= MW)
        model.addConstr(y[j] + y_prime[j] <= MH)
        model.addConstr(z[j] + z_prime[j] <= MD)

    # --- Capacity Weight ---
    for i in B:
        model.addConstr(gp.quicksum(item_data[j]['weight'] * p[item_to_customer[j], i] for j in I) <= CONFIG_GRB["vehicle_capacity_weight"])

    # --- Non-Overlapping ---
    for idx, j in enumerate(I):
        u_j = item_to_customer[j]
        for k in I[idx+1:]:
            u_k = item_to_customer[k]
            same_bin = model.addVar(vtype=GRB.BINARY)
            for i in B:
                model.addConstr(same_bin >= p[u_j, i] + p[u_k, i] - 1)
            
            model.addConstr(x[j] + x_prime[j] <= x[k] + MW * (1 - l_jk[j, k, 0]))
            model.addConstr(x[k] + x_prime[k] <= x[j] + MW * (1 - l_jk[j, k, 1]))
            model.addConstr(y[j] + y_prime[j] <= y[k] + MH * (1 - l_jk[j, k, 2]))
            model.addConstr(y[k] + y_prime[k] <= y[j] + MH * (1 - l_jk[j, k, 3]))
            model.addConstr(z[j] + z_prime[j] <= z[k] + MD * (1 - l_jk[j, k, 4]))
            model.addConstr(z[k] + z_prime[k] <= z[j] + MD * (1 - l_jk[j, k, 5]))
            model.addConstr(gp.quicksum(l_jk[j, k, a] for a in range(6)) >= same_bin)


    # --- MULTIDROP CONSTRAINTS (Substitusi LIFO) ---
    delta_j = 0 

    # Variabel L_prime[k, i]: Titik X terdalam (paling depan truk) yang ditempati pelanggan k
    L_prime = model.addVars(V_no_depot, B, lb=0, vtype=GRB.CONTINUOUS, name="L_prime")

    
    for i in B:
        for k in V_no_depot:
            # Eq 16: Batas L_prime tidak boleh melebihi panjang kontainer
            model.addConstr(L_prime[k, i] <= MW)

            # Eq 13: Menentukan nilai L_prime (Titik terjauh barang pelanggan k)
            # L_prime_ki >= x_j + x_prime_j untuk semua item j milik pelanggan k
            for j in I:
                if item_to_customer[j] == k:
                    model.addConstr(
                        L_prime[k, i] >= (x[j] + x_prime[j]) - MW * (1 - p[k, i]),
                        name=f"eq13_limit_{j}_{i}"
                    )

            # Eq 14: Aksesibilitas Multi-drop
            # Jika k dikunjungi SEBELUM l_next (f[k, l_next] = 1),
            # maka barang l_next harus dimulai SETELAH batas L_prime pelanggan k.
            for l_next in V_no_depot:
                if k == l_next: continue
                
                for j_next in I:
                    if item_to_customer[j_next] == l_next:
                        # Logika: x[j_next] >= L_prime[k, i]
                        # Ini memastikan barang pelanggan kedua tidak menghalangi barang pelanggan pertama (X=0)
                        model.addConstr(
                            x[j_next] >= (L_prime[k, i] - delta_j) - MW * (3 - f[k, l_next] - p[k, i] - p[l_next, i]),
                            name=f"eq14_access_{j_next}_{k}_{i}"
                        )

            # Eq 15: Sequence Limit (Hubungan L_prime antar pelanggan dalam rute)
            # Jika kendaraan i bergerak langsung dari k ke l_next (r[k, l_next, i] = 1)
            for l_next in V_no_depot:
                if k == l_next: continue
                model.addConstr(
                    L_prime[k, i] <= L_prime[l_next, i] + MW * (1 - r[k, l_next, i]),
                    name=f"eq15_sequence_{k}_{l_next}_{i}"
                )

    # --- CHAINING MUATAN ---
    for cust in V_no_depot:
        # 1. URUTKAN: Sangat penting agar solver membangun rantai yang logis
        cust_items = [j for j in I if item_to_customer[j] == cust]
        # Urutkan berdasarkan Panjang (X) agar balok yang lebih panjang jadi dasar
        cust_items.sort(key=lambda j: item_data[j]['length'], reverse=True)
        
        if len(cust_items) > 1:
            for i in range(1, len(cust_items)):
                item_prev = cust_items[i-1]
                item_curr = cust_items[i]
                
                # Variabel biner: 0 = Menempel di belakang (X), 1 = Menempel di atas (Z)
                stack_choice = model.addVar(vtype=GRB.BINARY, name=f"stack_choice_{cust}_{i}")
                
                # --- CEK FRAGILE: Tidak boleh ada barang di atas barang fragile ---
                is_prev_fragile = item_data[item_prev]['fragile']
                is_curr_fragile = item_data[item_curr]['fragile']
                
                # Jika item sebelumnya fragile, item sekarang WAJIB di sampingnya (X), bukan di atasnya
                if is_prev_fragile == 1:
                    model.addConstr(stack_choice == 0)

                # --- LOCK VEHICLE: Pastikan satu kendaraan (mencegah pisah truk) ---
                for v_idx in B:
                    model.addConstr(p[item_to_customer[item_curr], v_idx] == p[item_to_customer[item_prev], v_idx])

                # --- 1. KOORDINAT Y (LEBAR): Harus SAMA PERSIS ---
                # Ini menjamin barang berada dalam satu kolom/jalur yang sama
                model.addConstr(y[item_curr] == y[item_prev], name=f"coord_y_check_{cust}_{i}")

                # --- 2. CEK KOORDINAT X (HORIZONTAL) ---
                # Jika stack_choice = 0: x_curr HARUS TEPAT di ujung x_prev + x_prime_prev
                # Jika stack_choice = 1: x_curr HARUS SAMA dengan x_prev (tumpukan lurus)
                model.addConstr(x[item_curr] >= (x[item_prev] + x_prime[item_prev]) - MW * stack_choice, name=f"x_link_min_{i}")
                model.addConstr(x[item_curr] <= (x[item_prev] + x_prime[item_prev]) + MW * stack_choice, name=f"x_link_max_{i}")
                
                model.addConstr(x[item_curr] >= x[item_prev] - MW * (1 - stack_choice), name=f"x_align_min_{i}")
                model.addConstr(x[item_curr] <= x[item_prev] + MW * (1 - stack_choice), name=f"x_align_max_{i}")

                # --- 3. CEK KOORDINAT Z (VERTIKAL) ---
                # Jika stack_choice = 1: z_curr HARUS TEPAT di atas z_prev + z_prime_prev
                # Jika stack_choice = 0: z_curr HARUS SAMA dengan z_prev (sejajar lantai)
                model.addConstr(z[item_curr] >= (z[item_prev] + z_prime[item_prev]) - MD * (1 - stack_choice), name=f"z_link_min_{i}")
                model.addConstr(z[item_curr] <= (z[item_prev] + z_prime[item_prev]) + MD * (1 - stack_choice), name=f"z_link_max_{i}")

                model.addConstr(z[item_curr] >= z[item_prev] - MD * stack_choice, name=f"z_align_min_{i}")
                model.addConstr(z[item_curr] <= z[item_prev] + MD * stack_choice, name=f"z_align_max_{i}")

    # --- Stability dan Support ---
    for j in I:
        areas = []
        for r_idx in O:
            d = [item_data[j]['length'], item_data[j]['width'], item_data[j]['height']]
            areas.append(d[orientations[r_idx][0]] * d[orientations[r_idx][1]])
        
        base_area_j = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
        model.addConstr(base_area_j == gp.quicksum(areas[r_idx] * o[j, r_idx] for r_idx in O))

        ajb = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
        model.addConstr(ajb <= base_area_j)
        model.addConstr(ajb <= (MW * MH) * g[j])
        model.addConstr(z[j] <= MD * (1 - g[j]))

        v_kj_list = []
        for k in I:
            if j == k: continue
            v_kj = model.addVar(vtype=GRB.BINARY)
            v_kj_list.append(v_kj)
            model.addConstr(z[j] >= z[k] + z_prime[k] - MD * (1 - v_kj))
            model.addConstr(z[j] <= z[k] + z_prime[k] + MD * (1 - v_kj))
            model.addConstr(x[j] >= x[k] - MW * (1 - v_kj))
            model.addConstr(x[j] + x_prime[j] <= x[k] + x_prime[k] + MW * (1 - v_kj))
            model.addConstr(y[j] >= y[k] - MH * (1 - v_kj))
            model.addConstr(y[j] + y_prime[j] <= y[k] + y_prime[k] + MH * (1 - v_kj))

        model.addConstr(g[j] + gp.quicksum(v_kj_list) == 1)

    # --- Symmetry Breaking ---
    for cust in V_no_depot:
        cust_items = [i for i in I if item_to_customer[i] == cust]
        for idx in range(len(cust_items)-1):
            i1, i2 = cust_items[idx], cust_items[idx+1]
            if item_data[i1]['length'] == item_data[i2]['length'] and item_data[i1]['weight'] == item_data[i2]['weight']:
                model.addConstr(x[i1] <= x[i2])

    # ==========================================
    # 6. SOLVE
    # ==========================================
    model.optimize(gap_callback)

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        all_bins_data_grb = []
        total_distance_all_trucks = 0
        active_vehicles_count = 0

        for i in B:
            used_nodes = [v for v in V_no_depot if p[v, i].X > 0.5]
            
            if not used_nodes:
                continue

            active_vehicles_count += 1
            
            used_nodes.sort(key=lambda v: s[v].X)
            route_sequence = [depot_id] + used_nodes + [depot_id]

            bin_distance = 0
            bin_weight = 0
            bin_volume = 0
            for idx in range(len(route_sequence) - 1):
                bin_distance += dist(route_sequence[idx], route_sequence[idx+1])

            raw_items_in_bin = []
            for j in I:
                cust_j = item_to_customer[j]
                if p[cust_j, i].X > 0.5:
                    real_w = float(x_prime[j].X)
                    real_h = float(y_prime[j].X)
                    real_d = float(z_prime[j].X)
                    
                    item_info = {
                        "id": str(j),
                        "original_id": str(item_data[j]['id']),
                        "x": float(x[j].X), 
                        "y": float(y[j].X), 
                        "z": float(z[j].X),
                        "w": real_w, 
                        "h": real_h, 
                        "d": real_d,
                        "weight": float(item_data[j]['weight']),
                        "customer": str(cust_j),
                        "dest_x": float(coords[cust_j][0]),
                        "dest_y": float(coords[cust_j][1])
                    }
                    raw_items_in_bin.append(item_info)
                    bin_weight += item_data[j]['weight']
                    bin_volume += (real_w * real_h * real_d)

            node_order_map = {str(node): idx for idx, node in enumerate(route_sequence)}
            items_sorted = sorted(
                raw_items_in_bin, 
                key=lambda x: node_order_map.get(x['customer'], 999)
            )

            container_vol = bin_l * bin_w * bin_h
            volume_eff = (bin_volume / container_vol) * 100 if container_vol > 0 else 0
            total_distance_all_trucks += bin_distance

            all_bins_data_grb.append({
                "bin_id": int(i),
                "route": route_sequence,
                "items": items_sorted,
                "stats": {
                    "volume_eff": f"{volume_eff:.2f}%",
                    "total_weight": f"{float(bin_weight):.2f} kg",
                    "count": len(items_sorted),
                    "distance": f"{float(bin_distance):.2f} m",
                }
            })

        return {
            "success": True,
            "data": all_bins_data_grb,
            "stats": {
                "total_distance": total_distance_all_trucks,
                "vehicles_used": active_vehicles_count,
                "runtime": model.Runtime
            }
        }
    else:
        print("Optimasi berhenti: Tidak ditemukan solusi layak (Infeasible) atau error.")

def visualize_to_base64(df_results):
    history_file = 'results.csv'
    fig = None 
    
    try:
        file_exists = os.path.isfile(history_file)
        df_results.to_csv(history_file, mode='a', index=False, header=not file_exists)

        df_plot = pd.read_csv(history_file)
        
        datasets = df_plot['dataset_name'].tolist()
        container_sizes = df_plot['container_size'].tolist()

        j_gurobi = df_plot['dist_gurobi'].tolist()
        j_drl = df_plot['dist_drl'].tolist()
        
        w_gurobi = [max(0.01, v) for v in df_plot['time_gurobi']]
        w_drl = [max(0.01, v) for v in df_plot['time_drl']]

        gap_jarak = [((drl - gur) / gur * 100) if gur != 0 else 0 for drl, gur in zip(j_drl, j_gurobi)]
        speed_up = [(gur / drl) if drl != 0 else 0 for drl, gur in zip(w_drl, w_gurobi)]

        clean_datasets = ["_".join(ds.split('_')[2:]) if ds.count('_') >= 2 else ds for ds in datasets]
        new_labels = [f"{ds}\n({sz})\nRun-{i+1}" for i, (ds, sz) in enumerate(zip(clean_datasets, container_sizes))]
        
        x = np.arange(len(datasets))
        width = 0.35

        fig, axs = plt.subplots(2, 2, figsize=(20, 16), dpi=150)
        ((ax1, ax2), (ax3, ax4)) = axs

        # --- BARIS 1, KOLOM 1: Kualitas Jarak ---
        r1 = ax1.bar(x - width/2, j_gurobi, width, label='Gurobi', color='#2c3e50', alpha=0.8)
        r2 = ax1.bar(x + width/2, j_drl, width, label='MPPO', color='#e74c3c', alpha=0.8)
        ax1.set_title('Kualitas Solusi (Total Jarak)', fontsize=18, fontweight='bold', pad=15)
        ax1.set_ylabel('Jarak', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(new_labels, fontsize=12)
        ax1.legend()
        ax1.bar_label(r1, padding=3, fmt='%.2f', fontsize=12, rotation=90, fontweight='bold',)
        ax1.bar_label(r2, padding=3, fmt='%.2f', fontsize=12, rotation=90, fontweight='bold',)
        
        max_j = max(max(j_gurobi), max(j_drl))
        ax1.set_ylim(0, max_j * 1.4)

        # --- BARIS 1, KOLOM 2: Waktu Optimasi ---
        r3 = ax2.bar(x - width/2, w_gurobi, width, label='Gurobi', color='#2c3e50', alpha=0.8)
        r4 = ax2.bar(x + width/2, w_drl, width, label='MPPO', color='#27ae60', alpha=0.8)
        
        if len(datasets) > 1:
            try:
                for data_y, color, lbl in zip([w_gurobi, w_drl], ["#BFC243", "#ca3434"], ["Trend Gurobi", "Trend MPPO"]):
                    z = np.polyfit(x, data_y, 1)
                    p = np.poly1d(z)
                    ax2.plot(x, p(x), color=color, linestyle="--", linewidth=2, label=lbl, zorder=5)
            except:
                pass 

        ax2.set_yscale('log')
        ax2.set_title('Waktu Komputasi (Skala Log)', fontsize=18, fontweight='bold', pad=15)
        ax2.set_ylabel('Detik', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(new_labels, fontsize=12)
        
        ax2.legend(loc='upper left', fontsize=12)
        
        ax2.bar_label(r3, padding=3, fmt='%.2f', fontweight='bold', fontsize=12, rotation=90)
        ax2.bar_label(r4, padding=3, fmt='%.2f', fontweight='bold', fontsize=12, rotation=90)

        max_w = max(max(w_gurobi), max(w_drl))
        ax2.set_ylim(bottom=0.01, top=max_w * 100)

        # --- BARIS 2, KOLOM 1: Gap Jarak (%) ---
        r5 = ax3.bar(x, gap_jarak, width*1.2, color='#e67e22', alpha=0.7, edgecolor='black')
        ax3.set_title('Gap Jarak (%) terhadap Optimal', fontsize=18, fontweight='bold', pad=15)
        ax3.set_ylabel('Persentase (%)', fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels(new_labels, fontsize=12)
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.bar_label(r5, padding=3, fmt='%.2f%%', fontsize=12, fontweight='bold', rotation=90)
        ax3.set_ylim(0, max(gap_jarak + [5]) * 1.3)

        # --- BARIS 2, KOLOM 2: Gap Waktu (Speed-up Factor) ---
        r6 = ax4.bar(x, speed_up, width*1.2, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.set_title('Faktor Kecepatan Optimasi MPPO terhadap Optimal', fontsize=18, fontweight='bold', pad=15)
        ax4.set_ylabel('Kali Lebih Cepat', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(new_labels, fontsize=12)
        ax4.bar_label(r6, padding=3, fmt='%.1fx', fontsize=12, fontweight='bold', rotation=90)
        ax4.set_ylim(0, max(speed_up + [10]) * 1.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_base64

    except Exception as e:
        print(f"ERROR visualisasi: {str(e)}")
        return ""
    
    finally:
        if fig:
            plt.close(fig)
            plt.clf()

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
    print("Model MPPO Loaded Successfully")
except Exception as e:
    print(f"Error Loading Model: {e}")

@socketio.on('delete_results_file')
def handle_delete_file():
    file_path = 'results.csv'
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} berhasil dihapus.")
            emit('delete_status', {'status': 'success', 'message': 'File dihapus'})
        else:
            print("File tidak ditemukan.")
            emit('delete_status', {'status': 'error', 'message': 'File tidak ada'})
    except Exception as e:
        print(f"Gagal menghapus file: {str(e)}")
        emit('delete_status', {'status': 'error', 'message': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proses_data', methods=['POST'])
def proses_data():
    start_time = time.time()
    temp_path = "temp_manifest.csv"
    try:
        file = request.files['file']
        original_filename = file.filename
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

        drl_time = time.time() - start_time

        base_name = os.path.basename(original_filename)
        ds_name = os.path.splitext(base_name)[0]
        size_str = f"{int(bin_l)}x{int(bin_w)}x{int(bin_h)}"

        socketio.start_background_task(
            run_gurobi_background,
            temp_path, 
            ds_name, 
            size_str, 
            total_distance, 
            drl_time,
            depot_coords, bin_l, bin_h, bin_w
        )

        print("Waktu:", drl_time)
        print("Jarak", total_distance)

        return jsonify({
            "status": "success",
            "bins": all_bins_data,
            "bin_dims": [bin_l, bin_w, bin_h],
            "execution_time": int((drl_time) * 1000),
            "total_distance": f"{round(total_distance, 2)}",
            "message": "DRL optimization finished. Gurobi is running in background.",
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)