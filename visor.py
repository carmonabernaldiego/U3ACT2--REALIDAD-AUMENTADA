#!/usr/bin/env python3
"""
Visor simple de archivos OBJ para verificar que el modelo se carga correctamente
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_obj_file(filepath):
    """Cargar modelo 3D desde archivo .obj"""
    vertices = []
    faces = []
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):  # Vértice
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
                elif line.startswith('f '):  # Cara
                    parts = line.split()
                    face = []
                    for part in parts[1:]:
                        # Manejar diferentes formatos de caras (v, v/vt, v/vt/vn)
                        vertex_index = int(part.split('/')[0]) - 1  # OBJ usa índices base 1
                        face.append(vertex_index)
                    faces.append(face)
        
        print(f"✓ Modelo OBJ cargado exitosamente:")
        print(f"  - Vértices: {len(vertices)}")
        print(f"  - Caras: {len(faces)}")
        
        return np.array(vertices, dtype=np.float32), faces
        
    except FileNotFoundError:
        print(f"❌ Archivo {filepath} no encontrado")
        return None, None
    except Exception as e:
        print(f"❌ Error cargando archivo OBJ: {e}")
        return None, None

def analyze_model(vertices):
    """Analizar las propiedades del modelo"""
    if vertices is None or len(vertices) == 0:
        return
    
    print(f"\n📊 Análisis del modelo:")
    print(f"  - Número de vértices: {len(vertices)}")
    print(f"  - Rango X: {vertices[:, 0].min():.3f} a {vertices[:, 0].max():.3f}")
    print(f"  - Rango Y: {vertices[:, 1].min():.3f} a {vertices[:, 1].max():.3f}")
    print(f"  - Rango Z: {vertices[:, 2].min():.3f} a {vertices[:, 2].max():.3f}")
    
    # Calcular centro y dimensiones
    center = vertices.mean(axis=0)
    dimensions = vertices.max(axis=0) - vertices.min(axis=0)
    
    print(f"  - Centro: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  - Dimensiones: {dimensions[0]:.3f} x {dimensions[1]:.3f} x {dimensions[2]:.3f}")

def visualize_model(vertices, faces, title="Modelo 3D"):
    """Visualizar el modelo 3D usando matplotlib"""
    if vertices is None or len(vertices) == 0:
        print("❌ No hay datos para visualizar")
        return
    
    fig = plt.figure(figsize=(12, 8))
    
    # Vista 1: Puntos 3D
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=20)
    ax1.set_title('Vértices del modelo')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Vista 2: Wireframe
    ax2 = fig.add_subplot(222, projection='3d')
    
    # Dibujar wireframe usando las caras
    if faces and len(faces) > 0:
        for face in faces:
            if len(face) >= 3:
                for i in range(len(face)):
                    idx1 = face[i]
                    idx2 = face[(i + 1) % len(face)]
                    if idx1 < len(vertices) and idx2 < len(vertices):
                        x_line = [vertices[idx1][0], vertices[idx2][0]]
                        y_line = [vertices[idx1][1], vertices[idx2][1]]
                        z_line = [vertices[idx1][2], vertices[idx2][2]]
                        ax2.plot(x_line, y_line, z_line, 'b-', linewidth=0.5)
    else:
        # Si no hay caras, conectar puntos secuencialmente
        ax2.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'b-', linewidth=1)
    
    ax2.set_title('Wireframe del modelo')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Vista 3: Proyección XY
    ax3 = fig.add_subplot(223)
    ax3.scatter(vertices[:, 0], vertices[:, 1], c='green', s=10)
    ax3.set_title('Vista superior (XY)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True)
    
    # Vista 4: Proyección XZ
    ax4 = fig.add_subplot(224)
    ax4.scatter(vertices[:, 0], vertices[:, 2], c='purple', s=10)
    ax4.set_title('Vista frontal (XZ)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Z')
    ax4.grid(True)
    
    plt.suptitle(f'{title} - Thor Hammer', fontsize=16)
    plt.tight_layout()
    plt.show()

def test_obj_file(filepath):
    """Función principal para probar un archivo OBJ"""
    print(f"🔍 Analizando archivo: {filepath}")
    print("=" * 50)
    
    # Cargar modelo
    vertices, faces = load_obj_file(filepath)
    
    if vertices is not None:
        # Analizar modelo
        analyze_model(vertices)
        
        # Visualizar modelo
        try:
            visualize_model(vertices, faces, "Mjölnir")
        except Exception as e:
            print(f"⚠️  Error en visualización: {e}")
            print("Nota: Instala matplotlib para visualización: pip install matplotlib")
        
        # Mostrar algunos vértices de ejemplo
        print(f"\n📋 Primeros 10 vértices:")
        for i, vertex in enumerate(vertices[:10]):
            print(f"  v{i+1}: ({vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f})")
        
        if len(vertices) > 10:
            print(f"  ... y {len(vertices) - 10} vértices más")
        
        # Mostrar algunas caras de ejemplo
        if faces and len(faces) > 0:
            print(f"\n📐 Primeras 5 caras:")
            for i, face in enumerate(faces[:5]):
                print(f"  f{i+1}: {face}")
            if len(faces) > 5:
                print(f"  ... y {len(faces) - 5} caras más")
        else:
            print(f"\n⚠️  No se encontraron definiciones de caras")
        
        return True
    else:
        return False

if __name__ == "__main__":
    # Probar el archivo thor_hammer.obj
    obj_file = "thor_hammer.obj"
    
    print("🔨 Visor de archivo OBJ - Mjölnir")
    print("================================")
    
    success = test_obj_file(obj_file)
    
    if success:
        print(f"\n✅ El archivo {obj_file} se puede usar en la aplicación de AR")
    else:
        print(f"\n❌ Hay problemas con el archivo {obj_file}")
        print("\n💡 Sugerencias:")
        print("   - Verifica que el archivo existe en el directorio actual")
        print("   - Verifica que el archivo tiene formato OBJ válido")
        print("   - Prueba con otro modelo OBJ")
    
    print(f"\n🚀 Para usar en la app de AR, ejecuta: python marvel_ar_app.py")