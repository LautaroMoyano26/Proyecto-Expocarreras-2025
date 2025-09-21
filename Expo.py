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

# Sistema de puntuación por precisión
shots_fired = 0
shots_hit = 0
accuracy = 0.0

# Parámetros de los enemigos
ENEMY_WIDTH, ENEMY_HEIGHT = 40, 20
ENEMY_ROWS = 3
ENEMY_COLS = 8
ENEMY_GAP = 20
enemy_speed = 3
enemy_direction = 1

# Parámetros de disparo de enemigos
ENEMY_BULLET_FREQ = 0.002   # Probabilidad de disparo por enemigo por frame (menos disparos)
ENEMY_BULLET_SPEED = 8
enemy_bullets = []

# Niveles y jefe
level = 1
max_level = 3
jefe = None
jefe_bullets = []
jefe_speed = 8
jefe_hp = 20
player_hp = 5

# Patrones de disparo del jefe
jefe_shoot_timer = 0
jefe_pattern = 0  # 0=normal, 1=espiral, 2=90grados, 3=disperso
jefe_pattern_timer = 0
jefe_spiral_angle = 0

# Parámetros de disparos
bullet_width, bullet_height = 6, 18
bullets = []
bullet_speed = 10
can_shoot = True

# Limite de balas del jefe
MAX_JEFE_BULLETS = 30

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

def overlay_image_alpha(img, img_overlay, x, y):
    h, w = img_overlay.shape[:2]
    if x < 0:
        img_overlay = img_overlay[:, -x:]
        w = img_overlay.shape[1]
        x = 0
    if y < 0:
        img_overlay = img_overlay[-y:, :]
        h = img_overlay.shape[0]
        y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
        img_overlay = img_overlay[:, :w]
    if y + h > img.shape[0]:
        h = img.shape[0] - y
        img_overlay = img_overlay[:h, :]
    if w <= 0 or h <= 0:
        return
    if img_overlay.shape[2] == 4:
        alpha = img_overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            img[y:y+h, x:x+w, c] = alpha * img_overlay[:, :, c] + (1 - alpha) * img[y:y+h, x:x+w, c]
    else:
        img[y:y+h, x:x+w] = img_overlay

def crear_enemigos(level):
    enemies = []
    if level == 1:
        for row in range(ENEMY_ROWS):
            for col in range(ENEMY_COLS):
                x = 60 + col * (ENEMY_WIDTH + ENEMY_GAP)
                y = 60 + row * (ENEMY_HEIGHT + ENEMY_GAP)
                tipo = random.choice(['verde','naranja','rojo'])
                hp = 3 if tipo == 'rojo' else 2 if tipo == 'naranja' else 1
                enemies.append({'x':x, 'y':y, 'hp':hp})
    elif level == 2:
        center_x = WIDTH // 2
        start_y = 60
        rows = 5
        for row in range(rows):
            cols_in_row = row + 3
            start_x = center_x - (cols_in_row * (ENEMY_WIDTH + ENEMY_GAP)) // 2
            for col in range(cols_in_row):
                x = start_x + col * (ENEMY_WIDTH + ENEMY_GAP)
                y = start_y + row * (ENEMY_HEIGHT + ENEMY_GAP)
                tipo = random.choice(['verde','naranja','rojo'])
                hp = 3 if tipo == 'rojo' else 2 if tipo == 'naranja' else 1
                enemies.append({'x':x, 'y':y, 'hp':hp})
    else:
        center_x = WIDTH // 2
        center_y = 200
        num_enemies = 20
        for i in range(num_enemies):
            angle = (i * 2 * np.pi) / num_enemies
            radius = 80 + (i * 3)
            x = int(center_x + radius * np.cos(angle)) - ENEMY_WIDTH // 2
            y = int(center_y + radius * np.sin(angle)) - ENEMY_HEIGHT // 2
            x = max(0, min(WIDTH - ENEMY_WIDTH, x))
            y = max(60, min(HEIGHT // 2, y))
            tipo = random.choice(['verde','naranja','rojo'])
            hp = 3 if tipo == 'rojo' else 2 if tipo == 'naranja' else 1
            enemies.append({'x':x, 'y':y, 'hp':hp})
    return enemies

enemies = crear_enemigos(level)

def draw_enemies(frame, enemies):
    for e in enemies:
        sprite = enemigo3_img if e['hp']==3 else enemigo2_img if e['hp']==2 else enemigo1_img
        overlay_image_alpha(frame, sprite, e['x'], e['y'])

def draw_jefe(frame, jefe):
    if jefe:
        overlay_image_alpha(frame, ovni_img, jefe['x'], jefe['y'])
        cv2.putText(frame, f"Jefe: {jefe['hp']}", (jefe['x']+10, jefe['y']+ENEMY_HEIGHT), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 2)

def draw_bullets(frame, bullets):
    for b in bullets:
        x = b[0]
        y = b[1]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + bullet_width), int(y + bullet_height)), (0, 0, 255), -1)

def draw_ship(frame, x, y):
    overlay_image_alpha(frame, jugador_img, x, y)

def draw_enemy_bullets(frame, bullets):
    for b in bullets:
        x, y = b
        cv2.rectangle(frame, (int(x), int(y)), (int(x + bullet_width), int(y + bullet_height)), (0,255,255), -1)

def jefe_disparo_normal(jefe):
    return [[jefe['x'] + ENEMY_WIDTH, jefe['y'] + ENEMY_HEIGHT*2, 0, bullet_speed]]

def jefe_disparo_espiral(jefe, angle):
    bullets = []
    cx = jefe['x'] + ENEMY_WIDTH
    cy = jefe['y'] + ENEMY_HEIGHT*2
    for i in range(4):
        a = angle + i*np.pi/2
        dx = int(6*np.cos(a))
        dy = int(12 + abs(6*np.sin(a)))
        bullets.append([cx, cy, dx, dy])
    return bullets

def jefe_disparo_90grados(jefe):
    bullets = []
    cx = jefe['x'] + ENEMY_WIDTH
    cy = jefe['y'] + ENEMY_HEIGHT*2
    for dx, dy in [(0, bullet_speed), (-5, bullet_speed), (5, bullet_speed)]:
        bullets.append([cx, cy, dx, dy])
    return bullets

def jefe_disparo_disperso(jefe):
    bullets = []
    cx = jefe['x'] + ENEMY_WIDTH
    cy = jefe['y'] + ENEMY_HEIGHT*2
    for i in range(5):
        angle = np.pi/4 + i*np.pi/16
        dx = int(6*np.sin(angle))
        dy = int(10 + abs(6*np.cos(angle)))
        bullets.append([cx, cy, dx, dy])
    return bullets

def reset_game():
    global level, score, player_hp, jefe_hp, enemies, bullets, jefe, jefe_bullets, enemy_bullets, enemy_direction
    level = 1
    score = 0
    player_hp = 5
    jefe_hp = 20
    enemies = crear_enemigos(level)
    bullets = []
    jefe = None
    jefe_bullets = []
    enemy_bullets = []
    enemy_direction = 1

def show_menu(game_frame, options, selected):
    h = HEIGHT // 2 + 60
    for i, text in enumerate(options):
        color = (0,255,0) if i == selected else (255,255,255)
        cv2.putText(game_frame, text, (WIDTH//2-150, h + i*40), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

def main():
    global enemies, enemy_direction, bullets, can_shoot, ship_x, score, game_over, level
    global jefe, jefe_bullets, player_hp, jefe_hp, shots_fired, shots_hit, accuracy
    global jefe_shoot_timer, jefe_pattern, jefe_pattern_timer, jefe_spiral_angle, enemy_bullets
    global menu_selected, menu_active

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.2
    font_thickness = 2

    menu_active = False
    menu_options = []
    menu_selected = 0
    msg = ""

    # Mouse callback para el menú
    def mouse_callback(event, x, y, flags, param):
        global menu_selected, menu_active, game_over, shots_fired, shots_hit, accuracy
        if menu_active and event == cv2.EVENT_LBUTTONDOWN:
            h = HEIGHT // 2 + 60
            for i in range(len(menu_options)):
                if WIDTH//2-150 <= x <= WIDTH//2+250 and h + i*40 - 30 <= y <= h + i*40 + 30:
                    menu_selected = i
                    if menu_options[menu_selected] in ["Intentar otra vez", "Jugar de nuevo"]:
                        reset_game()
                        game_over = False
                        menu_active = False
                        shots_fired = 0
                        shots_hit = 0
                        accuracy = 0.0
                    elif menu_options[menu_selected] == "Salir":
                        cv2.destroyAllWindows()
                        exit()

    cv2.namedWindow('Space Invader')
    cv2.setMouseCallback('Space Invader', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cam_frame = cv2.resize(frame, (400, 300))
        cam_frame = cv2.flip(cam_frame, 1)
        frame_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_state = "No detectada"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(cam_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([pinky_tip.x, pinky_tip.y]))
                palm_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                ship_x = max(0, min(WIDTH - SHIP_WIDTH, int(palm_x*WIDTH) - SHIP_WIDTH//2))
                if dist > 0.3:
                    hand_state = "Mano abierta"
                    can_shoot = True
                else:
                    hand_state = "Mano cerrada"
                    if can_shoot and not menu_active:
                        bullets.append([ship_x + SHIP_WIDTH//2 - bullet_width//2, ship_y])
                        shots_fired += 1
                        can_shoot = False

        # --- PAUSA: Solo actualiza el juego si el menú NO está activo ---
        if not menu_active:
            # Mover enemigos
            for e in enemies:
                e['x'] += enemy_speed * enemy_direction
            if enemies:
                min_x = min([e['x'] for e in enemies])
                max_x = max([e['x'] for e in enemies])
                if max_x + ENEMY_WIDTH >= WIDTH or min_x <= 0:
                    enemy_direction *= -1
                    for e in enemies:
                        e['y'] += ENEMY_HEIGHT

            # Enemigos disparan en nivel 1 y 2
            if level in [1,2] and not jefe:
                for e in enemies:
                    if random.random() < ENEMY_BULLET_FREQ:
                        enemy_bullets.append([e['x'] + ENEMY_WIDTH//2, e['y'] + ENEMY_HEIGHT])

            # Mover balas del jugador
            for b in bullets:
                b[1] -= bullet_speed
            bullets = [b for b in bullets if b[1] > 0]

            # Mover balas del jefe
            new_jefe_bullets = []
            for b in jefe_bullets:
                if len(b) == 4:
                    b[0] += b[2]
                    b[1] += b[3]
                else:
                    b[1] += bullet_speed
                if 0 <= b[0] <= WIDTH and 0 <= b[1] <= HEIGHT:
                    new_jefe_bullets.append(b)
            jefe_bullets = new_jefe_bullets

            # Mover balas de enemigos
            for b in enemy_bullets:
                b[1] += ENEMY_BULLET_SPEED
            enemy_bullets = [b for b in enemy_bullets if b[1] < HEIGHT]

            # Colisiones con enemigos
            new_bullets = []
            for bullet in bullets:
                hit = False
                for e in enemies:
                    if (bullet[0] < e['x'] + ENEMY_WIDTH and bullet[0]+bullet_width > e['x'] and
                        bullet[1] < e['y'] + ENEMY_HEIGHT and bullet[1]+bullet_height > e['y']):
                        e['hp'] -= 1
                        if e['hp'] == 0: score += 50
                        shots_hit += 1
                        hit = True
                        break
                if not hit:
                    new_bullets.append(bullet)
            bullets = new_bullets
            enemies = [e for e in enemies if e['hp'] > 0]

            if shots_fired > 0:
                accuracy = (shots_hit / shots_fired)*100

            # Colisiones con jefe
            if jefe:
                new_bullets_jefe = []
                for bullet in bullets:
                    if (bullet[0] < jefe['x'] + ENEMY_WIDTH*2 and bullet[0]+bullet_width > jefe['x'] and
                        bullet[1] < jefe['y'] + ENEMY_HEIGHT*2 and bullet[1]+bullet_height > jefe['y']):
                        jefe['hp'] -= 1
                        shots_hit += 1
                        score += 200 if jefe['hp']==0 else 50
                    else:
                        new_bullets_jefe.append(bullet)
                bullets = new_bullets_jefe

            # Colisión balas del jefe con jugador
            for b in jefe_bullets[:]:
                bx, by = b[0], b[1]
                if (bx < ship_x + SHIP_WIDTH and bx + bullet_width > ship_x and
                    by < ship_y + SHIP_HEIGHT and by + bullet_height > ship_y):
                    player_hp -= 1
                    jefe_bullets.remove(b)

            # Colisión balas de enemigos con jugador
            for b in enemy_bullets[:]:
                bx, by = b[0], b[1]
                if (bx < ship_x + SHIP_WIDTH and bx + bullet_width > ship_x and
                    by < ship_y + SHIP_HEIGHT and by + bullet_height > ship_y):
                    player_hp -= 1
                    enemy_bullets.remove(b)

            # Control niveles y jefe
            if jefe:
                if jefe['hp'] <= 0:
                    level += 1
                    jefe = None
                    jefe_bullets = []
                    enemy_bullets = []
                    if level > max_level:
                        game_over = True
                        msg = "Victoria!"
                        menu_active = True
                        menu_options = ["Jugar de nuevo", "Salir"]
                        menu_selected = 0
                    else:
                        enemies = crear_enemigos(level)
                elif player_hp <= 0:
                    game_over = True
                    msg = "Derrota"
                    menu_active = True
                    menu_options = ["Intentar otra vez", "Salir"]
                    menu_selected = 0
            else:
                if not enemies:
                    level += 1
                    if level == max_level:
                        jefe = {'x': WIDTH//2-ENEMY_WIDTH, 'y': 60, 'hp': jefe_hp, "dir": 1}
                        jefe_bullets = []
                        enemy_bullets = []
                    elif level > max_level:
                        game_over = True
                        msg = "Victoria!"
                        menu_active = True
                        menu_options = ["Jugar de nuevo", "Salir"]
                        menu_selected = 0
                    else:
                        enemies = crear_enemigos(level)
                else:
                    for e in enemies:
                        if e['y'] + ENEMY_HEIGHT >= ship_y:
                            game_over = True
                            msg = "Derrota"
                            menu_active = True
                            menu_options = ["Intentar otra vez", "Salir"]
                            menu_selected = 0
                            break
            if player_hp <= 0 and not game_over:
                game_over = True
                msg = "Derrota"
                menu_active = True
                menu_options = ["Intentar otra vez", "Salir"]
                menu_selected = 0

            # Disparo y movimiento del jefe
            if jefe:
                jefe_shoot_timer += 1
                jefe_pattern_timer += 1
                if jefe_pattern_timer > 180:
                    jefe_pattern = (jefe_pattern + 1) % 4
                    jefe_pattern_timer = 0
                    jefe_spiral_angle = 0

                if len(jefe_bullets) < MAX_JEFE_BULLETS:
                    if jefe_pattern == 0 and jefe_shoot_timer > 80:
                        jefe_bullets.extend(jefe_disparo_normal(jefe))
                        jefe_shoot_timer = 0
                    elif jefe_pattern == 1 and jefe_shoot_timer > 20:
                        jefe_bullets.extend(jefe_disparo_espiral(jefe, jefe_spiral_angle))
                        jefe_spiral_angle += 0.4
                        jefe_shoot_timer = 0
                    elif jefe_pattern == 2 and jefe_shoot_timer > 100:
                        jefe_bullets.extend(jefe_disparo_90grados(jefe))
                        jefe_shoot_timer = 0
                    elif jefe_pattern == 3 and jefe_shoot_timer > 60:
                        jefe_bullets.extend(jefe_disparo_disperso(jefe))
                        jefe_shoot_timer = 0

                jefe['x'] += jefe_speed * jefe['dir']
                if jefe['x'] <= 0 or jefe['x'] + ENEMY_WIDTH*2 >= WIDTH:
                    jefe['dir'] *= -1

        # Dibujar frame
        game_frame = fondo_img.copy()
        draw_ship(game_frame, ship_x, ship_y)
        draw_enemies(game_frame, enemies)
        draw_bullets(game_frame, bullets)
        draw_jefe(game_frame, jefe)
        for b in jefe_bullets:
            x = b[0]
            y = b[1]
            cv2.rectangle(game_frame, (int(x), int(y)), (int(x + bullet_width), int(y + bullet_height)), (255,0,0), -1)
        draw_enemy_bullets(game_frame, enemy_bullets)

        # HUD
        cv2.putText(game_frame, f"Estado: {hand_state}", (10, 30), font, font_scale, (0,255,0), font_thickness)
        cv2.putText(game_frame, f"Puntaje: {score}", (10, 60), font, font_scale, (255,255,0), font_thickness)
        cv2.putText(game_frame, f"Nivel: {level}", (10, 90), font, font_scale, (255,0,255), font_thickness)
        cv2.putText(game_frame, f"HP: {player_hp}", (10, 120), font, font_scale, (0,128,255), font_thickness)
        cv2.putText(game_frame, f"Precision: {accuracy:.1f}%", (10, 150), font, font_scale, (255,255,255), font_thickness)
        cv2.putText(game_frame, f"Disparos: {shots_hit}/{shots_fired}", (10, 180), font, font_scale, (128,255,128), font_thickness)
        if jefe:
            pattern_names = ["Normal","Espiral","Cruz","Disperso"]
            cv2.putText(game_frame, f"Jefe: {pattern_names[jefe_pattern]}", (WIDTH-200, 30), font, font_scale, (255,128,0), font_thickness)

        if game_over:
            if game_over:
                color = (0,255,0) if msg == "Victoria!" else (0,0,255)
                cv2.putText(game_frame, msg, (WIDTH//2-150, HEIGHT//2), font, 2, color, 2)

        if menu_active:
            show_menu(game_frame, menu_options, menu_selected)

        cv2.imshow('Space Invader', game_frame)
        cv2.imshow('Camara', cam_frame)

        key = cv2.waitKey(1)
        if menu_active:
            if key == 2490368:  # flecha arriba
                menu_selected = (menu_selected - 1) % len(menu_options)
            elif key == 2621440:  # flecha abajo
                menu_selected = (menu_selected + 1) % len(menu_options)
            elif key == 13:  # enter
                if menu_options[menu_selected] in ["Intentar otra vez", "Jugar de nuevo"]:
                    reset_game()
                    game_over = False
                    menu_active = False
                    shots_fired = 0
                    shots_hit = 0
                    accuracy = 0.0
                elif menu_options[menu_selected] == "Salir":
                    break
        else:
            if key == 27 or game_over:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()