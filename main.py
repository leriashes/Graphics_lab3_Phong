import glfw
import glfw.GLFW as GLFWC
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr 
from PIL import Image

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 1

def initialize_glfw():
    glfw.init()
    glfw.window_hint(GLFWC.GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(GLFWC.GLFW_CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(GLFWC.GLFW_OPENGL_PROFILE, GLFWC.GLFW_OPENGL_CORE_PROFILE)
    glfw.window_hint(GLFWC.GLFW_OPENGL_FORWARD_COMPAT, GLFWC.GLFW_TRUE)
    glfw.window_hint(GLFWC.GLFW_DOUBLEBUFFER, GL_FALSE)

    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Lab3 Shevchenko", None, None)
    glfw.make_context_current(window)
    glfw.set_input_mode(window, GLFWC.GLFW_CURSOR, GLFWC.GLFW_CURSOR_HIDDEN)

    return window

class App:

    def __init__(self, window):
        self.window = window
        self.renderer = GraphicsEngine()
        self.scene = Scene()

        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            3: 45,
            4: 180,
            6: 135,
            7: 90,
            8: 270,
            9: 315,
            11: 0,
            12: 225,
            13: 270,
            14: 180
        }

        self.mainLoop()

    def mainLoop(self):
        running = True

        while(running):
            #проверка событий
            if glfw.window_should_close(self.window) or glfw.get_key(self.window, GLFWC.GLFW_KEY_ESCAPE) == GLFWC.GLFW_PRESS:
                running = False
            
            self.handleKeys()
            self.handleMouse()
            glfw.poll_events()
            self.scene.update(self.frameTime / 16.7)
            self.renderer.render(self.scene)
            self.calculateFramerate()
        self.quit() #выход из приложения

    def handleKeys(self):
        combo_move = 0
        directionModifier = 0

        if glfw.get_key(self.window, GLFWC.GLFW_KEY_W) == GLFWC.GLFW_PRESS:
            combo_move += 1
        if glfw.get_key(self.window, GLFWC.GLFW_KEY_A) == GLFWC.GLFW_PRESS:
            combo_move += 2
        if glfw.get_key(self.window, GLFWC.GLFW_KEY_S) == GLFWC.GLFW_PRESS:
            combo_move += 4
        if glfw.get_key(self.window, GLFWC.GLFW_KEY_D) == GLFWC.GLFW_PRESS:
            combo_move += 8
            
        if combo_move in self.walk_offset_lookup:
            directionModifier = self.walk_offset_lookup[combo_move]
            dPos = [
                0.5 * self.frameTime / 16.7 * np.cos(np.deg2rad(-self.scene.camera.phi + directionModifier)),
                0,
                -0.5 * self.frameTime / 16.7 * np.sin(np.deg2rad(-self.scene.camera.phi + directionModifier)),
            ]
            self.scene.move_camera(dPos)
            
    def handleMouse(self):
        (x, y) = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 16.7
        phi_increment = rate * ((SCREEN_WIDTH / 2) - x)
        theta_increment = rate * ((SCREEN_HEIGHT / 2) - y)
        self.scene.spin_camera(phi_increment, theta_increment)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def calculateFramerate(self):
        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = max(1, int(self.numFrames / delta))
            glfw.set_window_title(self.window, f"{framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0/max(1, framerate))
        self.numFrames += 1

    def quit(self):
        self.renderer.quit()

class GraphicsEngine:
    def __init__(self):

        self.color = (float(input("Введите цвет шара (R, G, B): ")) / 255, float(input()) / 255, float(input()) / 255)
        res = int(input("Введите частоту разбивки: "))

        self.sphere_mesh = SphereMesh(5, res)

        #инициализация opengl
        glClearColor(0.41, 0.41, 0.41, 1)  #цвет фона/очистки

        #создание шейдера
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")        
        glUseProgram(self.shader)

        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = SCREEN_WIDTH / SCREEN_HEIGHT,
            near = 0.1, far = 1000, dtype=np.float32
        )

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )

        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.lightLocation = {
            "position": [glGetUniformLocation(self.shader, f"Lights[{i}].position") for i in range(1)],
            "color": [glGetUniformLocation(self.shader, f"Lights[{i}].color") for i in range(1)],
            "strength": [glGetUniformLocation(self.shader, f"Lights[{i}].strength") for i in range(1)]
        }
        self.cameraPosLoc = glGetUniformLocation(self.shader, "cameraPosition")
        self.objColor = glGetUniformLocation(self.shader, "objcolor")

    def createShader(self, vertexFilepath, fragmentFilepath):
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader
    
    def render(self, scene):
        #обновление экрана
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)    #очистка экрана

        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye = scene.camera.position, 
            target = scene.camera.position + scene.camera.forwards,
            up = scene.camera.up, dtype=np.float32)
        
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

        for i, light in enumerate(scene.lights):
            glUniform3fv(self.lightLocation["position"][i], 1, light.position)
            glUniform3fv(self.lightLocation["color"][i], 1, light.color)
            glUniform1f(self.lightLocation["strength"][i], light.strength)
        glUniform3fv(self.cameraPosLoc, 1, scene.camera.position)
        glUniform3fv(self.objColor, 1, self.color)


        glBindVertexArray(self.sphere_mesh.vao)

        for sphere in scene.spheres:
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            #вращение
            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform, 
                m2 = pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(sphere.eulers), dtype=np.float32
                )
            )
            #передвижение
            model_transform = pyrr.matrix44.multiply(
                m1 = model_transform, 
                m2 = pyrr.matrix44.create_from_translation(
                    vec=np.array(sphere.position), dtype=np.float32
                )
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            glDrawArrays(GL_TRIANGLES, 0, self.sphere_mesh.vertex_count)

        glFlush()

    def quit(self):
        glDeleteProgram(self.shader)

class Camera:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.theta = 0
        self.phi = 0
        self.update_vectors()

    def update_vectors(self):
        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)),
                np.cos(np.deg2rad(self.theta)) * np.sin(np.deg2rad(self.phi)),
            ]
        )

        globalUp = np.array([0, 1, 0], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)

class Scene:
    def __init__(self):

        self.spheres = [
            Obj3D(
                position = [0, 0, 0],
                eulers = [0, 0, 0]
            ),
        ]

        self.lights = [
            Light(
                position=[8, 16, -8],
                color=[1, 1, 1],
                strength=100
            ),
        ]

        self.camera = Camera(position=[0, 5, 12])

    def update(self, rate):
        for cube in self.spheres:
            cube.eulers[0] += 0.25 * rate
            cube.eulers[1] += 0.25 * rate
            cube.eulers[2] += 0.25 * rate
            if cube.eulers[0] > 360:
                cube.eulers[0] -= 360
            if cube.eulers[1] > 360:
                cube.eulers[1] -= 360
            if cube.eulers[2] > 360:
                cube.eulers[2] -= 360

    def move_camera(self, dPos):
        dPos = np.array(dPos, dtype = np.float32)
        self.camera.position += dPos
    
    def spin_camera(self, dPhi, dTheta):
        self.camera.theta = min(89, max(-89, self.camera.theta + dTheta))
        self.camera.phi = (self.camera.phi - dPhi) % 360

        self.camera.update_vectors()

class Obj3D:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class Mesh:
    def __init__(self, filename):
        #x, y, z, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices) // 6
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        #вершины
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        #расположение
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        #нормаль
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def loadMesh(self, filename: str) -> list[float]:
        v = []
        vt = []
        vn = []

        vertices = []

        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                words = line.split(" ")
                if words[0] == "v":
                    v.append(self.read_vertex_data(words))
                elif words[0] == "vt":
                    vt.append(self.read_texcoord_data(words))
                elif words[0] == "vn":
                    vn.append(self.read_normal_data(words))
                elif words[0] == "f":
                    self.read_face_data(words, v, vt, vn, vertices)
                line = file.readline()

        return vertices
    
    def read_vertex_data(self, words: list[str]) -> list[float]:
        return [
            float(words[1]),
            float(words[2]),
            float(words[3])
        ]
    
    def read_texcoord_data(self, words: list[str]) -> list[float]:
        return [
            float(words[1]),
            float(words[2])
        ]

    def read_normal_data(self, words: list[str]) -> list[float]:
        return [
            float(words[1]),
            float(words[2]),
            float(words[3])
        ]

    def read_face_data(self, words: list[str],
                        v: list[list[float]], vt: list[list[float]], 
                        vn: list[list[float]], vertices: list[float]) -> None:
        triangleCount = len(words) - 3

        for i in range(triangleCount):
            self.make_corner(words[1], v, vt, vn, vertices)
            self.make_corner(words[2 + i], v, vt, vn, vertices)
            self.make_corner(words[3 + i], v, vt, vn, vertices)

    def make_corner(self, corner_description: str,
                    v: list[list[float]], vt: list[list[float]], 
                    vn: list[list[float]], vertices: list[float]) -> None:
        v_vt_vn = corner_description.split("/")
        if (v_vt_vn[0] != '\n'):
            for element in v[int(v_vt_vn[0]) - 1]:
                vertices.append(element)

            for element in vn[int(v_vt_vn[2]) - 1]:
                vertices.append(element)

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class SphereMesh(Mesh):
    def __init__(self, r, res, color=(255, 255, 255)):

        self.vertices = []
        self.faces = []
        self.color = color

        latitudes = [n * np.pi / res for n in range(1, res)] #вычисление "широт" (без полюсных)
        longitudes = [n * 2 * np.pi / res for n in range(res)] #вычисление "долгот"

        self.vertexes = [[r * np.sin(n) * np.sin(m), r * np.cos(m), r * np.cos(n) * np.sin(m)] for m in latitudes for n in longitudes]

        self.vertexes.append([0, -r, 0])
        self.vertexes.append([0, r, 0])

        normals = [[zn / r  for zn in vert]for vert in self.vertexes]
        

        num_nodes = (res - 1) * res

        #добавление граней (кроме полюсных)
        num_nodes = res * (res-1)
        self.addFaces([(m + n, 
                               (m + res) % num_nodes + n,
                               (m + res) % res**2 + (n + 1) % res)
                               for n in range(res) for m in range(0, num_nodes - res, res)])
        self.addFaces([(m + n, 
                               (m + res) % res**2 + (n + 1) % res,
                               m + (n + 1) % res) 
                               for n in range(res) for m in range(0, num_nodes - res, res)])

        #добавление полюсов и треугольных граней вокруг них
        self.addFaces([(n, (n + 1) % res, num_nodes + 1) for n in range(res)])
        start_node = num_nodes - res
        self.addFaces([(num_nodes, start_node + (n + 1) % res, start_node + n) for n in range(res)])

        for face in self.faces:
            for node in face[0]:
                for j in self.vertexes[node]:
                        self.vertices.append(j)
                for j in normals[node]:
                        self.vertices.append(j)


        self.vertex_count = len(self.vertices) // 6
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        #вершины
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        #расположение
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        #нормаль
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        

    def addFaces(self, face_list):
        for node_list in face_list:
            #num_nodes = len(node_list)
            if all((node < len(self.vertexes) for node in node_list)):
                self.faces.append([node_list, np.array(self.color, np.uint8)])

    

class Light:
    def __init__(self, position, color, strength):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

if __name__ == "__main__":
    
    window = initialize_glfw()
    myApp = App(window)