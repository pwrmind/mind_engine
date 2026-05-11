#!/usr/bin/env python3
"""
Mind Engine – фрактальный движок познания.
Консольная реализация концепции: Фокус, Сила незнания, рекурсия, Top-2, сатурация, базис.
Управление: Enter = шаг, s = шаг, a = авто, m = переключить режим, u = вверх, q = выход.
"""

import os
import random
import time
from typing import List, Optional, Tuple

# -------------------- Константы --------------------
MAX_DEPTH = 4          # Уровень «букв» (базис), глубже не идём
GRID_SIZE = 4          # Сетка 4x4 (16 подузлов)
SATURATION_DECAY = 1   # На сколько уменьшается сатурация за такт
SATURATION_ON_VISIT = 3
HISTORY_LENGTH = 8

# ANSI-цвета для консоли
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_GRAY = "\033[90m"

# -------------------- Класс Узла --------------------
class Node:
    def __init__(self, uid: str, parent: Optional['Node'] = None):
        self.uid = uid
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.is_flipped = False          # «монетка» перевёрнута (узел изучался)
        self.saturation = 0              # усталость от узла
        # Ленивое создание подузлов: None – ещё не создавались
        if self.depth < MAX_DEPTH:
            self.sub_nodes: List[Optional['Node']] = [None] * (GRID_SIZE * GRID_SIZE)
        else:
            self.sub_nodes = []          # базис – не имеет внутренностей

        # Глобальные счётчики (для прогресс-бара)
        global total_nodes_created, total_nodes_flipped
        total_nodes_created += 1
        if self.is_flipped:
            total_nodes_flipped += 1

    def get_child(self, index: int) -> 'Node':
        """Возвращает дочерний узел; создаёт при первом обращении."""
        if self.depth >= MAX_DEPTH:
            raise RuntimeError("Базис не может иметь детей.")
        if not 0 <= index < GRID_SIZE * GRID_SIZE:
            raise IndexError("Индекс за пределами сетки 4x4.")
        if self.sub_nodes[index] is None:
            self.sub_nodes[index] = Node(f"{self.uid}.{index}", self)
        return self.sub_nodes[index]

    def get_potential(self) -> float:
        """
        Потенциал незнания: 0 – всё известно, 1 – абсолютная неизвестность.
        Базис всегда 0; не перевёрнутый узел – 1;
        иначе доля неизвестных детей (с учётом ещё не созданных).
        """
        if self.depth >= MAX_DEPTH:
            return 0.0
        if not self.is_flipped:
            return 1.0
        unknown = 0
        for i in range(GRID_SIZE * GRID_SIZE):
            child = self.sub_nodes[i]
            if child is None:
                unknown += 1
            elif child.get_potential() > 0:
                unknown += 1
        return unknown / (GRID_SIZE * GRID_SIZE)

    def __repr__(self):
        return f"Node({self.uid})"

# -------------------- Глобальные счётчики --------------------
total_nodes_created = 0
total_nodes_flipped = 0

# -------------------- Главный движок --------------------
class MindEngine:
    def __init__(self):
        self.root = Node("root")
        self.root.is_flipped = True
        global total_nodes_flipped
        total_nodes_flipped += 1   # root считается изученным

        self.current_level = self.root
        self.focus_idx = 0         # индекс внутри current_level (0..15)
        self.tact = 0
        self.mode = "NORMAL"       # NORMAL или INVERSION
        self.history: List[int] = []   # история индексов для анти-цикла
        self.log_lines: List[str] = []

    # ---------- низкоуровневые операции ----------
    def _flip_node(self, node: Node):
        """Перевернуть «монетку» (изучить поверхностно)."""
        if not node.is_flipped:
            node.is_flipped = True
            global total_nodes_flipped
            total_nodes_flipped += 1

    def _saturation_decay(self):
        """Каждый такт усталость всех узлов на текущем уровне снижается."""
        if self.current_level.depth < MAX_DEPTH:
            for child in self.current_level.sub_nodes:
                if child and child.saturation > 0:
                    child.saturation -= SATURATION_DECAY

    # ---------- передвижение ----------
    def enter_node(self, idx: int):
        """Провалиться в узел на один уровень вглубь (с проверкой потенциала)."""
        node = self.current_level.get_child(idx)
        # Не входим, если узел уже полностью изучен (включая все вложенные слои)
        if node.get_potential() == 0.0:
            self._flip_node(node)   # на всякий случай перевернём, если ещё не
            self.add_log(f"Узел [{idx}] уже пуст, вход отменён.")
            return
        if node.depth >= MAX_DEPTH:
            self.add_log("Нельзя войти в базис (букву).")
            return

        self._flip_node(node)                  # поверхностное знакомство
        self.current_level = node
        self.focus_idx = 0
        self.history.clear()
        self.add_log(f">> Zoom IN: {node.uid}")

    def go_up(self):
        """Всплыть на родительский уровень."""
        if self.current_level.parent is None:
            return
        old_uid = self.current_level.uid
        self.current_level = self.current_level.parent
        # Пытаемся вернуть фокус на тот индекс, откуда пришли
        try:
            self.focus_idx = int(old_uid.split('.')[-1])
        except (ValueError, IndexError):
            self.focus_idx = 0
        self.history.clear()
        self.add_log(f"<< Zoom OUT: {self.current_level.uid}")

    # ---------- главный такт ----------
    def next_tact(self):
        self.tact += 1
        self._saturation_decay()

        focus_node = self.current_level.get_child(self.focus_idx)

        # ---- 0. Автоматический выход, если на слое нечего делать (Inversion) ----
        if self.mode == "INVERSION" and self.current_level.get_potential() == 0.0:
            if self.current_level.parent is not None:
                self.add_log("Слой полностью изучен, автоматический подъём.")
                self.go_up()
                return

        # ---- 1. Погружение вглубь (Inversion) ----
        if self.mode == "INVERSION" and focus_node.is_flipped and focus_node.get_potential() > 0:
            if focus_node.depth < MAX_DEPTH:
                self.enter_node(self.focus_idx)
                return

        # ---- 2. Изучение (переворот монетки) ----
        if not focus_node.is_flipped:
            self._flip_node(focus_node)
            self.add_log(f"Узел [{self.focus_idx}] открыт (flipped).")
            return

        # ---- 3. Горизонтальное перемещение ----
        candidates = []
        for i in range(GRID_SIZE * GRID_SIZE):
            if i == self.focus_idx:
                continue   # себя не рассматриваем
            node = self.current_level.get_child(i)
            pot = node.get_potential()

            # Базовый вес
            if self.mode == "NORMAL":
                weight = (1.0 - pot) * 10.0   # ищем знакомое
            else:
                weight = pot * 10.0           # ищем неизвестное

            # Штраф за историю
            try:
                hist_pos = self.history.index(i)
                weight -= 20.0 / (hist_pos + 1)
            except ValueError:
                pass

            # Штраф за сатурацию
            weight -= node.saturation * 5.0

            # Небольшой случайный шум (энтропия)
            weight += random.uniform(0.0, 1.5)

            candidates.append((i, weight))

        # Сортировка по убыванию веса
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_weight = candidates[0]

        # ---- 4. Автоматический выход наверх (Inversion) ----
        if self.mode == "INVERSION" and best_weight < 1.0 and self.current_level.parent is not None:
            self.add_log(f"Уровень исчерпан, всплываем...")
            self.go_up()
            return

        # ---- 5. Перемещение фокуса ----
        # Запоминаем историю
        self.history.insert(0, self.focus_idx)
        if len(self.history) > HISTORY_LENGTH:
            self.history.pop()

        self.focus_idx = best_idx
        # Увеличиваем сатурацию новому фокусному узлу
        new_focus = self.current_level.get_child(self.focus_idx)
        new_focus.saturation = SATURATION_ON_VISIT
        self.add_log(f"Фокус → [{self.focus_idx}] вес={best_weight:.1f}")

    # ---------- логирование ----------
    def add_log(self, msg: str):
        self.log_lines.append(f"Такт {self.tact}: {msg}")
        if len(self.log_lines) > 10:
            self.log_lines.pop(0)

    # ---------- визуализация ----------
    def render(self) -> str:
        """Возвращает строку с полным состоянием движка."""
        # Шапка
        global_percent = (total_nodes_flipped / total_nodes_created * 100) if total_nodes_created else 100
        lines = [
            f"{COLOR_BOLD}ТАКТ: {self.tact}{COLOR_RESET}   "
            f"РЕЖИМ: {COLOR_RED if self.mode == 'INVERSION' else COLOR_BLUE}{self.mode}{COLOR_RESET}   "
            f"ПРОГРЕСС: {global_percent:.1f}%",
            f"Уровень: {self.current_level.uid}   "
            f"Неопределённость слоя: {self.current_level.get_potential()*100:.0f}%",
            f"{COLOR_GRAY}─" * 40
        ]

        # Сетка 4x4
        for row in range(GRID_SIZE):
            row_str = ""
            for col in range(GRID_SIZE):
                idx = row * GRID_SIZE + col
                child = self.current_level.get_child(idx)
                flipped = child.is_flipped
                is_basis = child.depth >= MAX_DEPTH
                is_focus = (idx == self.focus_idx)
                pot = child.get_potential()

                # символ-индикатор
                if is_basis:
                    symbol = f"{COLOR_YELLOW}B{COLOR_RESET}"
                elif flipped:
                    symbol = f"{COLOR_BLUE}F{COLOR_RESET}"
                else:
                    symbol = f"{COLOR_GRAY}U{COLOR_RESET}"

                # полоска потенциала
                bar_len = 4
                filled_len = int((1.0 - pot) * bar_len)
                bar = f"[{COLOR_CYAN}{'#'*filled_len}{'.'*(bar_len-filled_len)}{COLOR_RESET}]"

                # рамка фокуса
                focus_border = ""
                if is_focus:
                    focus_border = f"{COLOR_RED if self.mode == 'INVERSION' else COLOR_CYAN}*{symbol}*{COLOR_RESET}"
                    cell = f"{focus_border}{bar}"
                else:
                    cell = f" {symbol} {bar}"

                row_str += cell + "  "
            lines.append(row_str)

        # Лог
        lines.append(f"{COLOR_GRAY}─" * 40)
        lines.append("Последние события:")
        for msg in self.log_lines[-6:]:
            lines.append(f"  {msg}")

        lines.append(f"{COLOR_GRAY}─" * 40)
        lines.append("[s] шаг  [a] авто  [m] режим  [u] вверх  [q] выход")
        return "\n".join(lines)

# -------------------- Интерактивная оболочка --------------------
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    engine = MindEngine()
    autoresume = False

    while True:
        clear_console()
        print(engine.render())
        if autoresume:
            # Авто-режим: сам идёт, пока не нажмут Ctrl+C
            try:
                while True:
                    time.sleep(0.2)
                    engine.next_tact()
                    clear_console()
                    print(engine.render())
            except KeyboardInterrupt:
                autoresume = False
                continue

        cmd = input("> ").strip().lower()
        if cmd in ('', 's'):
            engine.next_tact()
        elif cmd == 'a':
            autoresume = True
        elif cmd == 'm':
            engine.mode = "INVERSION" if engine.mode == "NORMAL" else "NORMAL"
            engine.add_log(f"Режим переключен на {engine.mode}")
        elif cmd == 'u':
            engine.go_up()
        elif cmd == 'q':
            print("Завершение работы.")
            break

if __name__ == "__main__":
    main()