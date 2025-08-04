#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarvelARApp v2.1 – Overlay 2D fix
Autor: Diego (con un empujón de ChatGPT)
Requisitos: OpenCV-Python ≥ 4.8, NumPy
"""

import cv2
import numpy as np
from cv2 import aruco
import os
from pathlib import Path
import time


class MarvelARApp:
    # ---------- CONFIGURACIÓN GENERAL ----------
    MARKER_SIZE = 6          # bits por lado del marcador
    TOTAL_MARKERS = 250
    CAPITAN_AMERICA_ID = 4   # Escudo (PNG)
    IRON_MAN_ID      = 5     # Sprite animado
    THOR_ID          = 6     # Mjölnir 3D

    def __init__(self):
        # Diccionario ArUco y parámetros
        self.aruco_dict = aruco.getPredefinedDictionary(
            getattr(aruco, f'DICT_{self.MARKER_SIZE}X{self.MARKER_SIZE}_{self.TOTAL_MARKERS}')
        )
        self.aruco_params = aruco.DetectorParameters()

        # Animación
        self.frame_index = 0
        self.animation_speed = 3
        self.animation_counter = 0

        # Recursos 2D/3D
        self.load_resources()

        # Matriz de cámara se re-calcula en run() con la resolución real
        self.camera_matrix = np.eye(3, dtype=np.float32)
        self.dist_coeffs   = np.zeros((4, 1))

        # Modelo 3D
        self.object_3d_points, self.mjolnir_faces = self.load_obj_file("thor_hammer.obj")
        self.mjolnir_face_normals = self.calculate_normals(self.object_3d_points, self.mjolnir_faces)

        # Luz frontal
        self.light_dir = np.array([0, 0, 1], dtype=np.float32)

        # Textura opcional de runas
        self.rune_texture = cv2.imread("runes.png", cv2.IMREAD_UNCHANGED) \
            if Path("runes.png").exists() else None

    # ---------- CARGA DE RECURSOS ----------
    def load_resources(self):
        # Escudo
        img = cv2.imread("capitan_america_shield.png", cv2.IMREAD_UNCHANGED)
        if img is None:
            self.capitan_america_shield = self.create_placeholder_image("ESCUDO", (300, 300), (0, 0, 255))
        else:
            # Asegura que tenga canal alfa
            self.capitan_america_shield = img if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Sprite Iron Man (4x2 frames)
        sprite = cv2.imread("iron_man_transformation.png", cv2.IMREAD_UNCHANGED)
        if sprite is not None:
            # Convierte a BGRA si hace falta
            sprite = sprite if sprite.shape[2] == 4 else cv2.cvtColor(sprite, cv2.COLOR_BGR2BGRA)
            self.iron_man_frames = self.split_sprite(sprite, cols=4, rows=2)
        else:
            self.iron_man_frames = self.create_placeholder_animation("IRON MAN", num_frames=8)

        print(f"Recursos cargados → Escudo: {self.capitan_america_shield.shape} | "
              f"Frames IronMan: {len(self.iron_man_frames)}")

    # ---------- PLACEHOLDERS ----------
    @staticmethod
    def create_placeholder_image(text, size, color):
        img = np.full((size[1], size[0], 4), (*color, 255), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.putText(img, text,
                    ((size[0] - ts[0]) // 2, (size[1] + ts[1]) // 2),
                    font, 1, (255, 255, 255), 2)
        return img

    def create_placeholder_animation(self, text, num_frames):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255),
                  (128, 128, 128), (255, 128, 0)]
        return [self.create_placeholder_image(f"{text} {i+1}", (200, 200),
                                              colors[i % len(colors)])
                for i in range(num_frames)]

    @staticmethod
    def split_sprite(sprite, cols, rows):
        h, w = sprite.shape[:2]
        fw, fh = w // cols, h // rows
        return [sprite[y*fh:(y+1)*fh, x*fw:(x+1)*fw]
                for y in range(rows) for x in range(cols)]

    # ---------- CARGA Y TRIANGULACIÓN DEL OBJ ----------
    def load_obj_file(self, filepath):
        if not Path(filepath).exists():
            print("⚠️  OBJ no encontrado; usando modelo low-poly por defecto.")
            return self.create_default_mjolnir_model()

        vertices, faces = [], []
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                if line.startswith('v '):
                    _, x, y, z = line.split()[:4]
                    vertices.append([float(x), float(y), float(z)])
                elif line.startswith('f '):
                    idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                    faces.append(idx)

        verts = np.array(vertices, dtype=np.float32)
        if verts.size == 0:
            print("⚠️  OBJ vacío; usando modelo de respaldo.")
            return self.create_default_mjolnir_model()

        verts -= verts.mean(axis=0)
        verts *= 2.0 / np.max(np.abs(verts))
        Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)
        verts = verts @ Rx.T

        faces_tri = []
        for f in faces:
            for i in range(1, len(f)-1):
                faces_tri.append([f[0], f[i], f[i+1]])

        print(f"✓ OBJ cargado: {len(verts)} vértices | {len(faces_tri)} caras (trianguladas)")
        return verts, faces_tri

    @staticmethod
    def create_default_mjolnir_model():
        pts = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],
               [-1.5, -0.8, 2.5], [1.5, -0.8, 2.5], [1.5, 0.8, 2.5], [-1.5, 0.8, 2.5],
               [-1.5, -0.8, 3.5], [1.5, -0.8, 3.5], [1.5, 0.8, 3.5], [-1.5, 0.8, 3.5]]
        faces = [[4,5,6], [4,6,7], [8,9,10], [8,10,11],
                 [4,5,9], [4,9,8], [5,6,10], [5,10,9],
                 [6,7,11], [6,11,10], [7,4,8], [7,8,11]]
        return np.array(pts, dtype=np.float32), faces

    @staticmethod
    def calculate_normals(verts, faces):
        normals = []
        for f in faces:
            v0, v1, v2 = verts[f]
            n = np.cross(v1 - v0, v2 - v0)
            n /= (np.linalg.norm(n) + 1e-8)
            normals.append(n.astype(np.float32))
        return np.array(normals)

    # ---------- DETECCIÓN ArUco ----------
    def find_aruco_markers(self, img, draw=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxes, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if draw and ids is not None:
            aruco.drawDetectedMarkers(img, bboxes, ids)
        return bboxes, ids

    # ---------- OVERLAY 2D CORREGIDO ----------
    def overlay_image_2d(self, img, marker_bbox, overlay_img):
        if overlay_img is None:
            return img

        # Asegura BGRA
        overlay = overlay_img if overlay_img.shape[2] == 4 else cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)

        corners = marker_bbox[0].reshape(4, 2).astype(np.float32)
        h, w = overlay.shape[:2]
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, corners)

        # Warp con bordes transparentes
        warped = cv2.warpPerspective(
            overlay, M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Alpha blend
        alpha = warped[:, :, 3] / 255.0
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - alpha) + warped[:, :, c] * alpha

        return img

    def overlay_animated_sprite(self, img, bbox, frames):
        frame = frames[self.frame_index % len(frames)]
        return self.overlay_image_2d(img, bbox, frame)

    # ---------- RENDER 3D (igual que v2.0) ----------
    def draw_obj_model(self, img, pts_2d, z_cam):
        light = self.light_dir / np.linalg.norm(self.light_dir)
        depth = np.array([z_cam[f].mean() for f in self.mjolnir_faces])
        order = depth.argsort()[::-1]

        for idx in order:
            poly = pts_2d[self.mjolnir_faces[idx]].astype(np.int32)
            if cv2.contourArea(poly) < 2.0: continue
            n    = self.mjolnir_face_normals[idx]
            inten= max(np.dot(n, light), 0.0)
            base = np.array([200, 200, 200], dtype=np.float32)
            accent = np.array([180, 180, 255], dtype=np.float32)
            color = (base * 0.7 + accent * 0.3 * inten).astype(np.uint8).tolist()
            cv2.fillConvexPoly(img, poly, color, lineType=cv2.LINE_AA)
        return img

    def draw_3d_object(self, img, marker_bbox):
        m = marker_bbox[0].reshape(4,2).astype(np.float32)
        real = 2.0
        marker_3d = np.array([[-real/2,-real/2,0],[ real/2,-real/2,0],
                              [ real/2, real/2,0],[-real/2, real/2,0]], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(marker_3d, m, self.camera_matrix,
                                      self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return img

        rotM, _ = cv2.Rodrigues(rvec)
        cam_pts = (rotM @ self.object_3d_points.T + tvec).T
        z_cam = cam_pts[:, 2]

        proj, _ = cv2.projectPoints(self.object_3d_points, rvec, tvec,
                                    self.camera_matrix, self.dist_coeffs)
        proj = proj.reshape(-1, 2)

        img = self.draw_obj_model(img, proj, z_cam)

        if self.rune_texture is not None:
            cx, cy = proj[self.mjolnir_faces[0]].mean(axis=0).astype(int)
            rh, rw = self.rune_texture.shape[:2]
            x1, x2 = cx - rw//4, cx + rw//4
            y1, y2 = cy - rh//4, cy + rh//4
            if 0 <= x1 < x2 <= img.shape[1] and 0 <= y1 < y2 <= img.shape[0]:
                rune = cv2.resize(self.rune_texture, (x2-x1, y2-y1))
                alpha= rune[:,:,3]/255.0
                for c in range(3):
                    img[y1:y2, x1:x2, c] = img[y1:y2, x1:x2, c] * (1-alpha) + rune[:,:,c] * alpha
        return img

    # ---------- LOOP PRINCIPAL ----------
    def process_frame(self, img):
        bboxes, ids = self.find_aruco_markers(img)
        if ids is not None:
            for bbox, mid in zip(bboxes, ids.flatten()):
                if   mid == self.CAPITAN_AMERICA_ID:
                    img = self.overlay_image_2d(img, bbox, self.capitan_america_shield)
                elif mid == self.IRON_MAN_ID:
                    img = self.overlay_animated_sprite(img, bbox, self.iron_man_frames)
                elif mid == self.THOR_ID:
                    img = self.draw_3d_object(img, bbox)
        self.animation_counter += 1
        if self.animation_counter >= self.animation_speed:
            self.frame_index += 1
            self.animation_counter = 0
        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_matrix[:] = [[w,0,w/2],[0,w,h/2],[0,0,1]]
        print("=== Marvel AR v2.1 ===", f"Resolución: {w}x{h}")
        print("Marcadores: 4=Escudo, 5=IronMan, 6=Mjölnir | ESC=Salir, R=Reset, S=Shot")
        shot = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            out = self.process_frame(frame)
            cv2.imshow("Marvel AR", out)
            k = cv2.waitKey(1) & 0xFF
            if   k == 27: break
            elif k == ord('r'): self.frame_index = self.animation_counter = 0
            elif k == ord('s'):
                fn = f"shot_{time.strftime('%Y%m%d_%H%M%S')}_{shot:02d}.png"
                cv2.imwrite(fn, out); print("Saved:", fn); shot+=1
        cap.release(); cv2.destroyAllWindows()


# ---------- GENERAR MARCADORES ----------
def generate_aruco_markers():
    ad = aruco.getPredefinedDictionary(
        getattr(aruco, f'DICT_{MarvelARApp.MARKER_SIZE}X{MarvelARApp.MARKER_SIZE}_{MarvelARApp.TOTAL_MARKERS}')
    )
    for mid, fn in [(MarvelARApp.CAPITAN_AMERICA_ID, "capitan_america_marker.png"),
                    (MarvelARApp.IRON_MAN_ID,       "iron_man_marker.png"),
                    (MarvelARApp.THOR_ID,           "thor_marker.png")]:
        cv2.imwrite(fn, aruco.generateImageMarker(ad, mid, 200))
        print(f"✓ {fn} (ID {mid})")
    print("¡Marcadores listos!")

if __name__ == "__main__":
    generate_aruco_markers()
    MarvelARApp().run()
