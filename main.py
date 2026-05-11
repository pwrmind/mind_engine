#!/usr/bin/env python3
"""
ResearchAgent v2.2 – агент-исследователь с раздельным хранением задач и базы знаний.
Сохраняет прогресс в tasks.md, ответы в knowledge.md.
Использует накопленный контекст при генерации подвопросов.
"""

import os
import re
import time
from typing import List, Optional, Dict, Set

# pip install ollama
from ollama import chat, list as ollama_list

# -------------------- Настройки --------------------
MAX_RESEARCH_DEPTH = 3
TASKS_FILE = "tasks.md"
KNOWLEDGE_FILE = "knowledge.md"
MODEL_NAME = "gemma4:e4b-it-q4_K_M"   # определится автоматически

# -------------------- Получение модели --------------------
def get_available_model() -> str:
    try:
        models = ollama_list()
        if not models:
            raise RuntimeError("Нет доступных моделей. Установите модель через `ollama pull <имя>`")
        name = models['model']
        print(f"Используется модель: {name}")
        return name
    except Exception as e:
        raise RuntimeError(f"Не удалось получить список моделей: {e}")

if MODEL_NAME is None:
    MODEL_NAME = get_available_model()

# -------------------- Узел задачи --------------------
class TaskNode:
    def __init__(self, question: str, parent: Optional['TaskNode'] = None, depth: int = 0,
                 is_closed: bool = False, answer: Optional[str] = None):
        self.question = question
        self.parent = parent
        self.depth = depth
        self.children: List[TaskNode] = []
        self.is_closed = is_closed
        self.answer = answer          # None, если открыт

    def add_child(self, child: 'TaskNode'):
        self.children.append(child)

    def close(self, answer: str):
        self.answer = answer
        self.is_closed = True

    def get_potential(self) -> float:
        """Потенциал незнания поддерева."""
        if self.is_closed and not self.children:
            return 0.0
        if self.is_closed:
            # узел закрыт, но могут быть открытые потомки
            child_pots = [c.get_potential() for c in self.children]
            return sum(child_pots) / len(child_pots)
        # узел открыт
        if not self.children:
            return 1.0
        child_pots = [c.get_potential() for c in self.children]
        return min(1.0, 0.5 + 0.5 * sum(child_pots) / len(child_pots))

    def count_open(self) -> int:
        n = 0 if self.is_closed else 1
        for c in self.children:
            n += c.count_open()
        return n

# -------------------- Парсер Markdown-дерева задач --------------------
def parse_tasks(markdown: str, parent_depth: int = 0, parent: Optional[TaskNode] = None) -> Optional[TaskNode]:
    """
    Восстанавливает дерево TaskNode из содержимого tasks.md.
    Возвращает корень дерева.
    """
    lines = markdown.splitlines()
    # Ищем первый заголовок первого уровня (# ...) и пропускаем его
    # Далее парсим строки вида "- [x] Вопрос" или "  - [ ] Вопрос"
    # Вложенность определяется количеством пробелов в начале строки (кратно 2).
    root = None
    stack = [( -1, None )]  # (indent_level, node)
    for line in lines:
        stripped = line.strip()
        # Пропускаем заголовок и пустые строки
        if stripped.startswith("#") or not stripped:
            continue
        # Определяем отступ (кратность 2 пробелам)
        indent = len(line) - len(line.lstrip(' '))
        level = indent // 2
        # Ищем чекбокс
        match = re.match(r"- \[([ x])\] (.*)", stripped)
        if not match:
            continue
        is_closed = (match.group(1) == 'x')
        question = match.group(2).strip()
        # Найти родителя по стеку: последний узел с уровнем < level
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_node = stack[-1][1] if stack else None
        node = TaskNode(question, parent=parent_node, depth=0 if parent_node is None else parent_node.depth + 1,
                        is_closed=is_closed)
        if parent_node:
            parent_node.add_child(node)
        if root is None:
            root = node
        stack.append((level, node))
    return root

def tasks_to_markdown(root: TaskNode, indent: int = 0) -> str:
    """Сериализует дерево задач в Markdown."""
    prefix = "  " * indent
    checkbox = "[x]" if root.is_closed else "[ ]"
    lines = [f"{prefix}- {checkbox} {root.question}"]
    for child in root.children:
        lines.append(tasks_to_markdown(child, indent + 1))
    return "\n".join(lines)

# -------------------- Работа с базой знаний (ответы) --------------------
def load_knowledge_base() -> Dict[str, str]:
    """Загружает ответы из knowledge.md (простой формат: ## Вопрос\n...\nОтвет:\n...)."""
    kb = {}
    if not os.path.exists(KNOWLEDGE_FILE):
        return kb
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    # Разбиваем по заголовкам второго уровня
    blocks = re.split(r"\n## ", content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Ищем "Ответ:" как разделитель
        parts = re.split(r"\nОтвет:\s*", block, maxsplit=1)
        question = parts[0].strip().lstrip("#").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        if question:
            kb[question] = answer
    return kb

def save_knowledge_base(kb: Dict[str, str]):
    """Сохраняет все ответы в knowledge.md."""
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        f.write("# База знаний\n\n")
        for q, a in kb.items():
            f.write(f"## {q}\n\nОтвет:\n{a}\n\n")

# -------------------- Агент --------------------
class ResearchAgent:
    def __init__(self, root_topic: str):
        self.root: Optional[TaskNode] = None
        self.knowledge_base: Dict[str, str] = load_knowledge_base()

        # Пытаемся загрузить дерево задач из tasks.md
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            self.root = parse_tasks(content)
            # Если есть закрытые задачи в дереве, но ответов нет в kb — добавим из file? Лучше не перегружать.
        if self.root is None:
            self.root = TaskNode(root_topic)
        self.tact = 0

    def save_state(self):
        """Сохраняет и задачи, и базу знаний."""
        if self.root:
            md = tasks_to_markdown(self.root)
            with open(TASKS_FILE, "w", encoding="utf-8") as f:
                f.write("# Дерево задач\n\n" + md)
        save_knowledge_base(self.knowledge_base)

    def find_most_uncertain(self, node: TaskNode) -> Optional[TaskNode]:
        candidates = []
        for child in node.children:
            cand = self.find_most_uncertain(child)
            if cand:
                candidates.append(cand)
        if not node.is_closed:
            candidates.append(node)
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.get_potential())

    def _build_context(self, node: TaskNode) -> str:
        """Собирает краткий контекст из ответов родительской цепочки и ближайших известных."""
        context = []
        # поднимаемся к корню, собираем ответы
        current = node.parent
        while current:
            if current.answer and current.is_closed:
                # Берем первые 150 символов ответа
                snippet = current.answer.strip()[:200].replace("\n", " ")
                context.append(f"- {current.question}: {snippet}...")
            current = current.parent
        # Добавим также ответы на братские закрытые вопросы (одного родителя)
        if node.parent:
            for sibling in node.parent.children:
                if sibling is not node and sibling.is_closed and sibling.answer:
                    snippet = sibling.answer.strip()[:200].replace("\n", " ")
                    context.append(f"- {sibling.question}: {snippet}...")
        if not context:
            return "Пока нет известных фактов."
        return "Уже известны следующие факты:\n" + "\n".join(context[-5:])  # последние 5 фактов

    def generate_subquestions(self, node: TaskNode) -> List[str]:
        context_text = self._build_context(node)
        prompt = (
            f"Ты исследователь. Тема: «{node.question}».\n"
            f"{context_text}\n"
            "На основе этих фактов, сформулируй 3-5 самых важных уточняющих подвопросов, "
            "которые ещё не освещены. Верни каждый вопрос с новой строки, без нумерации."
        )
        system_msg = "Ты – эксперт по структурированию знаний. Отвечай строго по формату."
        response = chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': prompt}
            ]
        )
        text = response['message']['content'].strip()
        questions = []
        for line in text.splitlines():
            line = line.strip("-*•1234567890. ").strip()
            if line and len(line) > 5:
                questions.append(line)

        # Фильтрация дубликатов
        existing = set(self.knowledge_base.keys())
        # Добавим все вопросы из всего дерева (чтобы не повторяться)
        def collect_questions(n: TaskNode):
            nonlocal existing
            existing.add(n.question)
            for c in n.children:
                collect_questions(c)
        if self.root:
            collect_questions(self.root)
        filtered = [q for q in questions if q not in existing]
        return filtered[:5]

    def research(self, node: TaskNode):
        if node.question in self.knowledge_base:
            node.close(self.knowledge_base[node.question])
            print(f"  [из кэша] {node.question}")
            return

        prompt = f"Дай развёрнутый ответ на вопрос: «{node.question}». Приведи примеры. Ответь на русском."
        system_msg = "Ты – полезный помощник для исследований."
        response = chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': prompt}
            ]
        )
        answer = response['message']['content'].strip()
        node.close(answer)
        self.knowledge_base[node.question] = answer
        print(f"  [изучено] {node.question}")

        if node.depth < MAX_RESEARCH_DEPTH:
            subqs = self.generate_subquestions(node)
            for q in subqs:
                child = TaskNode(q, parent=node, depth=node.depth + 1)
                node.add_child(child)
            if subqs:
                print(f"  -> порождено {len(subqs)} подвопросов (с учётом контекста)")

    def step(self) -> bool:
        self.tact += 1
        open_cnt = self.root.count_open() if self.root else 0
        print(f"\n--- Такт {self.tact} (открыто задач: {open_cnt}) ---")
        target = self.find_most_uncertain(self.root)
        if not target:
            print("Исследование завершено.")
            self.build_final_report()
            return False
        print(f"Цель: {target.question}")
        self.research(target)
        self.save_state()
        return True

    def build_final_report(self):
        lines = ["# Финальный отчёт", f"Тема: {self.root.question}", ""]
        def collect(n: TaskNode):
            if n.is_closed and n.answer:
                lines.append(f"- **{n.question}**: {n.answer}")
            for c in n.children:
                collect(c)
        if self.root:
            collect(self.root)
        report = "\n".join(lines)
        with open("final_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("Финальный отчёт сохранён в final_report.md")

    def auto_research(self, max_tacts=50):
        for _ in range(max_tacts):
            if not self.step():
                break
            time.sleep(1.0)

# -------------------- Интерфейс --------------------
def main():
    topic = input("Введите тему для исследования: ").strip()
    if not topic:
        print("Тема не задана.")
        return
    agent = ResearchAgent(topic)
    print(f"Агент готов. Тем: {topic}")
    print("[s] шаг  [a] авто 10  [r] отчёт  [q] выход")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ('s', ''):
            agent.step()
        elif cmd == 'a':
            agent.auto_research(10)
        elif cmd == 'r':
            agent.build_final_report()
            print(agent.final_report if hasattr(agent, 'final_report') else "Нет данных")
        elif cmd == 'q':
            agent.save_state()
            print("Прогресс сохранён. Выход.")
            break
        else:
            print("Неизвестная команда")

if __name__ == "__main__":
    main()