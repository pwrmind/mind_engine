#!/usr/bin/env python3
"""
ResearchAgent – агент-исследователь на основе метода «Силы незнания».
Строит дерево задач, сохраняет базу знаний в Markdown, использует локальный LLM (Ollama)
для рассуждений, переиспользует накопленные факты.

Разделены: база знаний (словарь + файл) и список задач (дерево).
Поиск самой «незнаемой» задачи исправлен — обход не прерывается на закрытых узлах.
"""

import os
import re
import time
from typing import List, Optional, Dict

# pip install ollama
from ollama import chat, list as ollama_list

# -------------------- Настройки --------------------
MAX_RESEARCH_DEPTH = 3                # максимальная глубина дерева задач
KNOWLEDGE_FILE = "knowledge_base.md"  # файл базы знаний
MODEL_NAME = "gemma4:e4b-it-q4_K_M"                     # будет определена автоматически

# -------------------- Функция получения доступной модели Ollama --------------------
def get_available_model() -> str:
    """Возвращает имя первой доступной модели Ollama, а если нет – завершает работу с ошибкой."""
    try:
        models = ollama_list()
        if not models:
            raise RuntimeError("Нет доступных моделей Ollama. Установите хотя бы одну, например: ollama pull llama2")
        # Берём первую модель из списка
        model_name = models['model']
        print(f"Используется модель: {model_name}")
        return model_name
    except Exception as e:
        raise RuntimeError(f"Не удалось получить список моделей Ollama. Убедитесь, что Ollama запущена. Ошибка: {e}")

if MODEL_NAME is None:
    MODEL_NAME = get_available_model()

# -------------------- Узел задачи (TaskNode) --------------------
class TaskNode:
    def __init__(self, question: str, parent: Optional['TaskNode'] = None, depth: int = 0):
        self.question = question
        self.parent = parent
        self.depth = depth
        self.children: List[TaskNode] = []
        self.answer: Optional[str] = None   # None = ещё не исследована
        self.is_closed = False

    def add_child(self, child: 'TaskNode'):
        self.children.append(child)

    def close(self, answer: str):
        self.answer = answer
        self.is_closed = True

    def get_potential(self) -> float:
        """Вычисляет потенциал незнания поддерева. Если узел закрыт, его собственный потенциал 0, но учитываем детей."""
        if self.is_closed:
            # Узел закрыт, но его потомки могут быть открыты
            if not self.children:
                return 0.0
            # Потенциал = средний потенциал детей
            child_potentials = [c.get_potential() for c in self.children]
            return sum(child_potentials) / len(child_potentials)
        else:
            # Открытый узел: его собственный потенциал 1 + учёт детей
            if not self.children:
                return 1.0
            child_pot = sum(c.get_potential() for c in self.children) / len(self.children)
            return min(1.0, 0.5 + 0.5 * child_pot)

    def count_open(self) -> int:
        """Количество незакрытых задач в поддереве (включая себя)."""
        count = 0 if self.is_closed else 1
        for child in self.children:
            count += child.count_open()
        return count

# -------------------- Агент --------------------
class ResearchAgent:
    def __init__(self, root_topic: str):
        self.root = TaskNode(root_topic)
        self.knowledge_base: Dict[str, str] = {}   # вопрос -> ответ
        self.load_knowledge_base()
        self.tact = 0
        self.final_report = ""

    def load_knowledge_base(self):
        """Загрузить существующие ответы из Markdown."""
        if not os.path.exists(KNOWLEDGE_FILE):
            return
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        # Ищем строки: - [x] Вопрос\n  - Ответ: ...
        pattern = r"- \[x\] (.*?)\n  - Ответ: (.*?)(?=\n- \[|$)"
        for match in re.finditer(pattern, content, re.DOTALL):
            q = match.group(1).strip()
            a = match.group(2).strip()
            self.knowledge_base[q] = a
        if self.knowledge_base:
            print(f"Загружено {len(self.knowledge_base)} известных ответов из базы.")

    def save_knowledge_base(self):
        """Сохраняет базу знаний в Markdown, обходя дерево задач."""
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            f.write(f"# Исследование: {self.root.question}\n\n")
            f.write(self._node_to_markdown(self.root))
            f.write("\n\n---\n")
            f.write(f"Тактов исследования: {self.tact}\n")

    def _node_to_markdown(self, node: TaskNode, indent: int = 0) -> str:
        """Рекурсивно преобразует поддерево в Markdown-список."""
        prefix = "  " * indent
        checkbox = "[x]" if node.is_closed else "[ ]"
        lines = [f"{prefix}- {checkbox} {node.question}"]
        if node.answer:
            answer_text = node.answer.replace("\n", "\n" + prefix + "    ")
            lines.append(f"{prefix}  - Ответ: {answer_text}")
        for child in node.children:
            lines.append(self._node_to_markdown(child, indent + 1))
        return "\n".join(lines)

    def find_most_uncertain(self, node: TaskNode) -> Optional[TaskNode]:
        """
        Рекурсивно находит узел с наивысшим потенциалом незнания.
        ВАЖНО: обходит всех потомков, даже если сам узел закрыт.
        """
        # Сначала собираем кандидатов из потомков
        candidates = []
        for child in node.children:
            candidate = self.find_most_uncertain(child)
            if candidate:
                candidates.append(candidate)
        # Если текущий узел открыт, он тоже кандидат
        if not node.is_closed:
            candidates.append(node)

        if not candidates:
            return None
        # Возвращаем узел с максимальным потенциалом (чем выше, тем больше незнания)
        return max(candidates, key=lambda n: n.get_potential())

    def generate_subquestions(self, node: TaskNode) -> List[str]:
        """
        С помощью LLM генерирует 3-5 подвопросов для раскрытия темы.
        Исключает вопросы, уже присутствующие в knowledge_base или среди прямых детей.
        """
        prompt = (
            f"Ты исследователь. Тема: «{node.question}».\n"
            "Какие 3-5 самых важных подвопросов нужно изучить, чтобы полностью понять эту тему?\n"
            "Верни каждый вопрос с новой строки, без нумерации, только сами вопросы."
        )
        system_msg = "Ты – эксперт по структурированию знаний. Отвечай строго по формату."
        response = chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': prompt}
            ]
        )
        response_text = response['message']['content'].strip()

        questions = []
        for line in response_text.splitlines():
            line = line.strip("-*•1234567890. ").strip()
            if line and len(line) > 5:
                questions.append(line)

        # Фильтруем: не добавляем те, что уже есть в базе знаний или среди существующих детей
        existing = set(self.knowledge_base.keys()) | {child.question for child in node.children}
        filtered = [q for q in questions if q not in existing]
        if not filtered:
            # Если LLM не дал новых вопросов, добавляем стандартные
            fallback = [f"Что такое {node.question}?", f"Как работает {node.question}?"]
            filtered = [q for q in fallback if q not in existing]
        return filtered

    def research(self, node: TaskNode):
        """Выполнить исследование одного узла."""
        # Проверяем кэш (базу знаний)
        if node.question in self.knowledge_base:
            node.close(self.knowledge_base[node.question])
            print(f"  [из кэша] {node.question}")
            return

        # Запрашиваем ответ у LLM
        prompt = (
            f"Дай развёрнутый, но чёткий ответ на вопрос: «{node.question}».\n"
            "Если возможно, приведи примеры. Ответь на русском языке."
        )
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

        # Генерируем подзадачи, если не на максимальной глубине
        if node.depth < MAX_RESEARCH_DEPTH:
            subqs = self.generate_subquestions(node)
            for q in subqs:
                child = TaskNode(q, parent=node, depth=node.depth + 1)
                node.add_child(child)
            if subqs:
                print(f"  -> порождено {len(subqs)} подвопросов")

    def step(self):
        """Один такт работы агента. Возвращает True, если есть ещё работа, иначе False."""
        self.tact += 1
        open_count = self.root.count_open()
        print(f"\n--- Такт {self.tact} (осталось открытых задач: {open_count}) ---")

        target = self.find_most_uncertain(self.root)
        if not target:
            print("Все задачи исследованы. Исследование завершено.")
            self.build_final_report()
            return False

        print(f"Исследуем: {target.question}")
        self.research(target)
        self.save_knowledge_base()
        return True

    def build_final_report(self):
        """Генерирует финальный отчёт на основе закрытых задач."""
        lines = [
            f"# Финальный отчёт по теме: {self.root.question}",
            "",
            "Собраны следующие факты:",
        ]
        def collect_answers(node: TaskNode):
            if node.is_closed and node.answer:
                lines.append(f"- **{node.question}**: {node.answer}")
            for child in node.children:
                collect_answers(child)
        collect_answers(self.root)
        self.final_report = "\n".join(lines)
        with open("final_report.md", "w", encoding="utf-8") as f:
            f.write(self.final_report)
        print("\nФинальный отчёт сохранён в final_report.md")

    def auto_research(self, max_tacts: int = 100):
        """Автоматический режим до исчерпания задач или лимита тактов."""
        for _ in range(max_tacts):
            if not self.step():
                break
            time.sleep(1.0)

# -------------------- Интерфейс --------------------
def main():
    topic = input("Введите тему для исследования: ").strip()
    if not topic:
        print("Тема не введена, выход.")
        return
    agent = ResearchAgent(topic)
    print(f"Агент запущен. Тема: {topic}")
    print("Команды: s - шаг, a - авто 10 тактов, q - выход, r - отчёт")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ('s', ''):
            agent.step()
        elif cmd == 'a':
            agent.auto_research(max_tacts=10)
        elif cmd == 'r':
            agent.build_final_report()
            print(agent.final_report)
        elif cmd == 'q':
            print("Завершение. База знаний сохранена.")
            break
        else:
            print("Неизвестная команда.")

if __name__ == "__main__":
    main()