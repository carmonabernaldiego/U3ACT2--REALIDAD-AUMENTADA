import cv2
import numpy as np
import math

def create_captain_america_shield():
    """Crear un escudo del Capitán América básico"""
    size = 300
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    # Fondo transparente
    img[:, :, 3] = 0
    
    # Círculos concéntricos
    radii = [140, 120, 100, 80, 60, 40]
    colors = [(0, 0, 255, 255), (255, 255, 255, 255), (255, 0, 0, 255), 
              (255, 255, 255, 255), (255, 0, 0, 255), (0, 0, 255, 255)]
    
    for radius, color in zip(radii, colors):
        cv2.circle(img, center, radius, color, -1)
    
    # Estrella central
    star_points = []
    for i in range(5):
        angle = i * 2 * math.pi / 5 - math.pi / 2
        outer_x = center[0] + int(30 * math.cos(angle))
        outer_y = center[1] + int(30 * math.sin(angle))
        star_points.append([outer_x, outer_y])
        
        angle = (i + 0.5) * 2 * math.pi / 5 - math.pi / 2
        inner_x = center[0] + int(15 * math.cos(angle))
        inner_y = center[1] + int(15 * math.sin(angle))
        star_points.append([inner_x, inner_y])
    
    star_points = np.array(star_points, np.int32)
    cv2.fillPoly(img, [star_points], (255, 255, 255, 255))
    
    return img

def create_iron_man_sprite():
    """Crear sprite animado de Iron Man (simulado)"""
    frame_width, frame_height = 200, 200
    cols, rows = 4, 2
    total_width = frame_width * cols
    total_height = frame_height * rows
    
    sprite = np.zeros((total_height, total_width, 4), dtype=np.uint8)
    
    colors = [
        (0, 0, 200, 255),    # Azul oscuro
        (0, 50, 220, 255),   # Azul medio
        (50, 100, 255, 255), # Azul claro
        (100, 150, 255, 255),# Azul muy claro
        (255, 100, 100, 255),# Rojo claro
        (255, 50, 50, 255),  # Rojo medio
        (255, 0, 0, 255),    # Rojo
        (255, 215, 0, 255)   # Dorado (Iron Man)
    ]
    
    for row in range(rows):
        for col in range(cols):
            frame_idx = row * cols + col
            x_start = col * frame_width
            y_start = row * frame_height
            x_end = x_start + frame_width
            y_end = y_start + frame_height
            
            # Fondo del frame
            sprite[y_start:y_end, x_start:x_end, :] = colors[frame_idx]
            
            # Forma básica de Iron Man (rectángulo con "ojos")
            center_x = x_start + frame_width // 2
            center_y = y_start + frame_height // 2
            
            # Cuerpo
            cv2.rectangle(sprite, 
                         (center_x - 40, center_y - 60), 
                         (center_x + 40, center_y + 60), 
                         (255, 215, 0, 255), -1)
            
            # "Ojos" de Iron Man
            cv2.circle(sprite, (center_x - 15, center_y - 20), 8, (255, 255, 255, 255), -1)
            cv2.circle(sprite, (center_x + 15, center_y - 20), 8, (255, 255, 255, 255), -1)
            
            # Reactor arc (pecho)
            cv2.circle(sprite, (center_x, center_y + 10), 12, (0, 255, 255, 255), -1)
            
            # Número de frame
            cv2.putText(sprite, f"{frame_idx + 1}", 
                       (x_start + 10, y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
    
    return sprite

def create_placeholder_images():
    """Crear todas las imágenes placeholder necesarias"""
    print("Creando imágenes placeholder para Marvel AR...")
    
    # Crear escudo del Capitán América
    shield = create_captain_america_shield()
    cv2.imwrite("capitan_america_shield.png", shield)
    print("✓ Creado: capitan_america_shield.png")
    
    # Crear sprite de Iron Man
    iron_man_sprite = create_iron_man_sprite()
    cv2.imwrite("iron_man_transformation.png", iron_man_sprite)
    print("✓ Creado: iron_man_transformation.png")
    
    print("¡Imágenes placeholder creadas exitosamente!")
    print("Puedes reemplazarlas con imágenes de mejor calidad si las tienes.")

if __name__ == "__main__":
    create_placeholder_images()