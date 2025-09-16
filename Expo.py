import cv2
import numpy as np
import mediapipe as mp
import random

# Parámetros de la ventana
WIDTH, HEIGHT = 1080, 720

# Parámetros de la nave
SHIP_WIDTH, SHIP_HEIGHT = 60, 30
ship_x = WIDTH // 2 - SHIP_WIDTH // 2
ship_y = HEIGHT - SHIP_HEIGHT - 10
score = 0
game_over = False

# Parámetros de los enemigos
ENEMY_WIDTH, ENEMY_HEIGHT = 40, 20
ENEMY_ROWS = 3
ENEMY_COLS = 8
ENEMY_GAP = 20
enemy_speed = 3
enemy_direction = 1

# Niveles y jefe
level = 1
max_level = 3
jefe = None
jefe_bullets = []
jefe_speed = 8
jefe_hp = 20
player_hp = 5

# Parámetros de disparos
bullet_width, bullet_height = 6, 18
bullets = []
bullet_speed = 10
can_shoot = True

# Cargar sprites y redimensionar
enemigo1_img = cv2.imread('enemigo1.png', cv2.IMREAD_UNCHANGED)
enemigo1_img = cv2.resize(enemigo1_img, (ENEMY_WIDTH, ENEMY_HEIGHT), interpolation=cv2.INTER_AREA)

enemigo2_img = cv2.imread('enemigo2.png', cv2.IMREAD_UNCHANGED)
enemigo2_img = cv2.resize(enemigo2_img, (ENEMY_WIDTH, ENEMY_HEIGHT), interpolation=cv2.INTER_AREA)

enemigo3_img = cv2.imread('enemigo3.png', cv2.IMREAD_UNCHANGED)
enemigo3_img = cv2.resize(enemigo3_img, (ENEMY_WIDTH, ENEMY_HEIGHT), interpolation=cv2.INTER_AREA)

ovni_img = cv2.imread('ovni.png', cv2.IMREAD_UNCHANGED)
ovni_img = cv2.resize(ovni_img, (ENEMY_WIDTH*2, ENEMY_HEIGHT*2), interpolation=cv2.INTER_AREA)

jugador_img = cv2.imread('jugador1.png', cv2.IMREAD_UNCHANGED)
jugador_img = cv2.resize(jugador_img, (ENEMY_WIDTH*2, ENEMY_HEIGHT*2), interpolation=cv2.INTER_AREA)

# Cargar fondo
fondo_img = cv2.imread('fondo.png', cv2.IMREAD_UNCHANGED)
fondo_img = cv2.resize(fondo_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

# Función para superponer imágenes con transparencia
def overlay_image_alpha(img, img_overlay, x, y):
    h, w = img_overlay.shape[:2]
    if x + w > img.shape[1] or y + h > img.shape[0]:
        return
    if img_overlay.shape[2] == 4:  # tiene canal alpha
        alpha = img_overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            img[y:y+h, x:x+w, c] = alpha * img_overlay[:, :, c] + (1 - alpha) * img[y:y+h, x:x+w, c]
    else:
        img[y:y+h, x:x+w] = img_overlay

# Crear enemigos
def crear_enemigos(level):
    enemies = []
    for row in range(ENEMY_ROWS):
        for col in range(ENEMY_COLS):
            x = 60 + col * (ENEMY_WIDTH + ENEMY_GAP)
            y = 60 + row * (ENEMY_HEIGHT + ENEMY_GAP)
            tipo = random.choice(['verde','naranja','rojo'])
            if tipo == 'rojo':
                hp = 3
            elif tipo == 'naranja':
                hp = 2
            else:
                hp = 1
            enemies.append({'x':x, 'y':y, 'hp':hp})
    return enemies

enemies = crear_enemigos(level)

# Funciones para dibujar
def draw_enemies(frame, enemies):
    for e in enemies:
        if e['hp'] == 3:
            sprite = enemigo3_img
        elif e['hp'] == 2:
            sprite = enemigo2_img
        else:
            sprite = enemigo1_img
        overlay_image_alpha(frame, sprite, e['x'], e['y'])

def draw_jefe(frame, jefe):
    if jefe:
        overlay_image_alpha(frame, ovni_img, jefe['x'], jefe['y'])
        cv2.putText(frame, f"Jefe: {jefe['hp']}", (jefe['x']+10, jefe['y']+ENEMY_HEIGHT), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

def draw_bullets(frame, bullets):
    for (x, y) in bullets:
        cv2.rectangle(frame, (x, y), (x + bullet_width, y + bullet_height), (0, 0, 255), -1)

def draw_ship(frame, x, y):
    overlay_image_alpha(frame, jugador_img, x, y)

def main():
    global enemies, enemy_direction, bullets, can_shoot, ship_x, score, game_over, level, jefe, jefe_bullets, player_hp, jefe_hp
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Configuración de fuente pixelada
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cam_frame = cv2.resize(frame, (400, 300))
        cam_frame = cv2.flip(cam_frame, 1)
        frame_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_state = "No detectada"
        hand_x = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(cam_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([pinky_tip.x, pinky_tip.y]))
                palm_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                hand_x = int(palm_x * WIDTH)
                ship_x = max(0, min(WIDTH - SHIP_WIDTH, hand_x - SHIP_WIDTH // 2))
                if dist > 0.3:
                    hand_state = "Mano abierta"
                    can_shoot = True
                else:
                    hand_state = "Mano cerrada"
                    if can_shoot:
                        bullets.append([ship_x + SHIP_WIDTH // 2 - bullet_width // 2, ship_y])
                        can_shoot = False

        # Mover enemigos
        for i in range(len(enemies)):
            enemies[i]['x'] += enemy_speed * enemy_direction
        if enemies:
            min_x = min([e['x'] for e in enemies])
            max_x = max([e['x'] for e in enemies])
            if max_x + ENEMY_WIDTH >= WIDTH or min_x <= 0:
                enemy_direction *= -1
                for i in range(len(enemies)):
                    enemies[i]['y'] += ENEMY_HEIGHT

        # Mover balas
        for i in range(len(bullets)):
            bullets[i][1] -= bullet_speed
        bullets = [b for b in bullets if b[1] > 0]

        for i in range(len(jefe_bullets)):
            jefe_bullets[i][1] += bullet_speed
        jefe_bullets = [b for b in jefe_bullets if b[1] < HEIGHT]

        # Colisiones
        new_bullets = []
        for bullet in bullets:
            hit = False
            for e in enemies:
                if (bullet[0] < e['x'] + ENEMY_WIDTH and bullet[0] + bullet_width > e['x'] and
                    bullet[1] < e['y'] + ENEMY_HEIGHT and bullet[1] + bullet_height > e['y'] and e['hp'] > 0):
                    e['hp'] -= 1
                    if e['hp'] == 0:
                        score += 10
                    hit = True
                    break
            if not hit:
                new_bullets.append(bullet)
        bullets = new_bullets
        enemies = [e for e in enemies if e['hp'] > 0]

        if jefe:
            new_bullets_jefe = []
            for bullet in bullets:
                if (bullet[0] < jefe['x'] + ENEMY_WIDTH*2 and bullet[0] + bullet_width > jefe['x'] and
                    bullet[1] < jefe['y'] + ENEMY_HEIGHT*2 and bullet[1] + bullet_height > jefe['y'] and jefe['hp'] > 0):
                    jefe['hp'] -= 1
                    if jefe['hp'] == 0:
                        score += 100
                else:
                    new_bullets_jefe.append(bullet)
            bullets = new_bullets_jefe

        for bx, by in jefe_bullets:
            if (bx < ship_x + SHIP_WIDTH and bx + bullet_width > ship_x and
                by < ship_y + SHIP_HEIGHT and by + bullet_height > ship_y):
                player_hp -= 1
                jefe_bullets.remove([bx, by])

        if jefe:
            if jefe['hp'] <= 0:
                level += 1
                jefe = None
                jefe_bullets = []
                if level > max_level:
                    game_over = True
                    msg = "¡Victoria!"
                else:
                    enemies = crear_enemigos(level)
            elif player_hp <= 0:
                game_over = True
                msg = "Derrota"
        else:
            if not enemies:
                level += 1
                if level == max_level:
                    jefe = {'x': WIDTH//2-ENEMY_WIDTH, 'y': 60, 'hp': jefe_hp}
                    jefe_bullets = []
                elif level > max_level:
                    game_over = True
                    msg = "¡Victoria!"
                else:
                    enemies = crear_enemigos(level)
            else:
                for e in enemies:
                    if e['y'] + ENEMY_HEIGHT >= ship_y:
                        game_over = True
                        msg = "Derrota"
                        break

        # Frame del juego con fondo
        game_frame = fondo_img.copy()
        draw_ship(game_frame, ship_x, ship_y)
        draw_enemies(game_frame, enemies)
        draw_bullets(game_frame, bullets)
        draw_jefe(game_frame, jefe)
        for bx, by in jefe_bullets:
            cv2.rectangle(game_frame, (bx, by), (bx + bullet_width, by + bullet_height), (255,0,0), -1)

        # HUD estilo pixelado 
        cv2.putText(game_frame, f"Estado: {hand_state}", (10, 30), font, font_scale, (0,255,0), font_thickness, cv2.LINE_8)
        cv2.putText(game_frame, f"Puntaje: {score}", (10, 60), font, font_scale, (255,255,0), font_thickness, cv2.LINE_8)
        cv2.putText(game_frame, f"Nivel: {level}", (10, 90), font, font_scale, (255,0,255), font_thickness, cv2.LINE_8)
        cv2.putText(game_frame, f"HP: {player_hp}", (10, 120), font, font_scale, (0,128,255), font_thickness, cv2.LINE_8)
        if game_over:
            cv2.putText(game_frame, msg, (WIDTH//2-150, HEIGHT//2), font, 4, (0,0,255), 4, cv2.LINE_8)

        if jefe and np.random.rand() < 0.03:
            jefe_bullets.append([jefe['x']+ENEMY_WIDTH, jefe['y']+ENEMY_HEIGHT*2])
        if jefe:
            jefe['x'] += jefe_speed * enemy_direction
            if jefe['x'] <= 0 or jefe['x']+ENEMY_WIDTH*2 >= WIDTH:
                enemy_direction *= -1

        cv2.imshow('Space Invader', game_frame)
        cv2.imshow('Camara', cam_frame)
        key = cv2.waitKey(1)
        if key == 27 or game_over:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
