#!/usr/bin/env python3
"""
ollama_client.py
================
Скрипт для отправки запросов к локальному серверу Ollama через HTTP API.

Использование:
    python ollama_client.py [ОПЦИИ] -p "ваш промпт"

Опции:
    -p, --prompt  <str>  Промпт для модели                    [обязательный]
    -m, --model   <str>  Название модели                      [по умолчанию: qwen2.5:0.5b]
    -u, --url     <str>  URL сервера Ollama                   [по умолчанию: http://localhost:11434]
    -s, --stream         Включить потоковый вывод             [по умолчанию: выключен]
    -r, --raw            Вывести сырой JSON ответ             [по умолчанию: выключен]

Примеры:
    python ollama_client.py -p "Что такое Linux?"
    python ollama_client.py -p "Напиши Hello World" -m qwen2.5:0.5b
    python ollama_client.py -p "Объясни Docker" --stream
    python ollama_client.py -p "Привет" --raw
    python ollama_client.py -p "Привет" -u http://192.168.1.10:11434

Зависимости:
    pip install requests

Коды возврата:
    0 — успех
    1 — ошибка аргументов
    2 — ошибка соединения с сервером
    3 — ошибка парсинга ответа
"""

import argparse
import json
import sys
from typing import Generator

import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

DEFAULT_MODEL   = "qwen2.5:0.5b"
DEFAULT_URL     = "http://localhost:11434"
API_GENERATE    = "/api/generate"
API_TAGS        = "/api/tags"
TIMEOUT_CONNECT = 5    # секунд на установку соединения
TIMEOUT_READ    = 120  # секунд на чтение ответа


# =============================================================================
# ЦВЕТА
# =============================================================================

class Color:
    """ANSI-коды для цветного вывода в терминале."""

    RED    = "\033[0;31m"
    GREEN  = "\033[0;32m"
    CYAN   = "\033[0;36m"
    YELLOW = "\033[1;33m"
    RESET  = "\033[0m"

    @staticmethod
    def red(text: str) -> str:
        """Обернуть текст в красный цвет."""
        return f"{Color.RED}{text}{Color.RESET}"

    @staticmethod
    def green(text: str) -> str:
        """Обернуть текст в зелёный цвет."""
        return f"{Color.GREEN}{text}{Color.RESET}"

    @staticmethod
    def cyan(text: str) -> str:
        """Обернуть текст в голубой цвет."""
        return f"{Color.CYAN}{text}{Color.RESET}"


# =============================================================================
# ЛОГГЕР
# =============================================================================

class Logger:
    """Простой логгер для вывода сообщений в stderr."""

    @staticmethod
    def info(message: str) -> None:
        """Вывести информационное сообщение."""
        print(Color.cyan("[INFO]"), message, file=sys.stderr)

    @staticmethod
    def error(message: str) -> None:
        """Вывести сообщение об ошибке."""
        print(Color.red("[ERROR]"), message, file=sys.stderr)

    @staticmethod
    def separator() -> None:
        """Вывести разделитель."""
        print("-" * 40, file=sys.stderr)


log = Logger()


# =============================================================================
# КЛИЕНТ OLLAMA
# =============================================================================

class OllamaClient:
    """
    HTTP клиент для взаимодействия с Ollama API.

    Attributes:
        base_url (str): Базовый URL сервера Ollama.
        model    (str): Название используемой модели.
        session  (requests.Session): Переиспользуемая HTTP сессия.
    """

    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> None:
        """
        Инициализация клиента.

        Args:
            base_url: Базовый URL сервера Ollama.
            model:    Название модели для запросов.
        """
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # -------------------------------------------------------------------------

    def check_server(self) -> None:
        """
        Проверяет доступность сервера Ollama.

        Делает GET запрос на /api/tags — стандартный health-check endpoint.

        Raises:
            SystemExit(2): Если сервер недоступен или не отвечает.
        """
        log.info(f"Проверяю доступность сервера: {self.base_url}")

        try:
            response = self.session.get(
                url     = f"{self.base_url}{API_TAGS}",
                timeout = TIMEOUT_CONNECT,
            )
            response.raise_for_status()

        except ConnectionError:
            log.error(f"Сервер недоступен: {self.base_url}")
            log.error("Запустите сервер командой: ollama serve")
            sys.exit(2)

        except ReadTimeout:
            log.error("Сервер не отвечает (таймаут)")
            sys.exit(2)

        log.info("Сервер доступен ✓")

    # -------------------------------------------------------------------------

    def _build_payload(self, prompt: str, stream: bool) -> dict:
        """
        Собирает тело запроса для API Ollama.

        Args:
            prompt: Текст запроса пользователя.
            stream: Включить потоковый вывод.

        Returns:
            Словарь с полями model, prompt, stream.
        """
        return {
            "model":  self.model,
            "prompt": prompt,
            "stream": stream,
        }

    # -------------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Отправляет запрос и возвращает полный текстовый ответ модели.

        Args:
            prompt: Текст запроса пользователя.

        Returns:
            Полный текстовый ответ модели в виде строки.

        Raises:
            SystemExit(2): При ошибке соединения.
            SystemExit(3): При ошибке парсинга ответа.
        """
        payload = self._build_payload(prompt, stream=False)

        try:
            response = self.session.post(
                url     = f"{self.base_url}{API_GENERATE}",
                json    = payload,
                timeout = (TIMEOUT_CONNECT, TIMEOUT_READ),
            )
            response.raise_for_status()

        except ConnectionError as e:
            log.error(f"Ошибка соединения: {e}")
            sys.exit(2)

        except ReadTimeout:
            log.error(f"Таймаут ожидания ответа ({TIMEOUT_READ}с)")
            sys.exit(2)

        except RequestException as e:
            log.error(f"Ошибка запроса: {e}")
            sys.exit(2)

        try:
            return response.json()["response"]

        except (KeyError, json.JSONDecodeError) as e:
            log.error(f"Ошибка парсинга ответа: {e}")
            log.error(f"Сырой ответ: {response.text[:200]}")
            sys.exit(3)

    # -------------------------------------------------------------------------

    def generate_raw(self, prompt: str) -> dict:
        """
        Отправляет запрос и возвращает полный JSON ответ сервера.

        Args:
            prompt: Текст запроса пользователя.

        Returns:
            Словарь с полным JSON ответом от Ollama.

        Raises:
            SystemExit(2): При ошибке соединения.
            SystemExit(3): При ошибке парсинга ответа.
        """
        payload = self._build_payload(prompt, stream=False)

        try:
            response = self.session.post(
                url     = f"{self.base_url}{API_GENERATE}",
                json    = payload,
                timeout = (TIMEOUT_CONNECT, TIMEOUT_READ),
            )
            response.raise_for_status()
            return response.json()

        except ConnectionError as e:
            log.error(f"Ошибка соединения: {e}")
            sys.exit(2)

        except ReadTimeout:
            log.error(f"Таймаут ожидания ответа ({TIMEOUT_READ}с)")
            sys.exit(2)

        except json.JSONDecodeError as e:
            log.error(f"Ошибка парсинга JSON: {e}")
            sys.exit(3)

    # -------------------------------------------------------------------------

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Отправляет запрос и возвращает генератор токенов (потоковый режим).

        Сервер возвращает ответ построчно в формате NDJSON.
        Каждая строка — отдельный JSON объект с полем 'response'.

        Args:
            prompt: Текст запроса пользователя.

        Yields:
            Отдельные токены (части ответа) по мере их поступления.

        Raises:
            SystemExit(2): При ошибке соединения.
            SystemExit(3): При ошибке парсинга токена.
        """
        payload = self._build_payload(prompt, stream=True)

        try:
            response = self.session.post(
                url     = f"{self.base_url}{API_GENERATE}",
                json    = payload,
                timeout = (TIMEOUT_CONNECT, TIMEOUT_READ),
                stream  = True,
            )
            response.raise_for_status()

        except ConnectionError as e:
            log.error(f"Ошибка соединения: {e}")
            sys.exit(2)

        except ReadTimeout:
            log.error(f"Таймаут ожидания ответа ({TIMEOUT_READ}с)")
            sys.exit(2)

        # Читаем ответ построчно
        for raw_line in response.iter_lines():
            if not raw_line:
                continue

            try:
                chunk = json.loads(raw_line)

            except json.JSONDecodeError as e:
                log.error(f"Ошибка парсинга токена: {e}")
                sys.exit(3)

            yield chunk.get("response", "")

            # Генерация завершена
            if chunk.get("done", False):
                break


# =============================================================================
# ПАРСИНГ АРГУМЕНТОВ
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    Returns:
        Namespace с полями: prompt, model, url, stream, raw.
    """
    parser = argparse.ArgumentParser(
        prog            = "ollama_client.py",
        description     = "Клиент для отправки запросов к Ollama API",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = "Пример: python ollama_client.py -p 'Что такое Linux?'",
    )

    parser.add_argument(
        "-p", "--prompt",
        type     = str,
        required = True,
        help     = "Промпт (вопрос) для модели",
    )
    parser.add_argument(
        "-m", "--model",
        type    = str,
        default = DEFAULT_MODEL,
        help    = f"Название модели (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-u", "--url",
        type    = str,
        default = DEFAULT_URL,
        help    = f"URL сервера Ollama (по умолчанию: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-s", "--stream",
        action  = "store_true",
        default = False,
        help    = "Включить потоковый вывод",
    )
    parser.add_argument(
        "-r", "--raw",
        action  = "store_true",
        default = False,
        help    = "Вывести сырой JSON ответ",
    )

    return parser.parse_args()


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main() -> None:
    """
    Точка входа в скрипт.

    Алгоритм:
        1. Разобрать аргументы командной строки.
        2. Создать клиент OllamaClient.
        3. Проверить доступность сервера.
        4. Отправить запрос в нужном режиме (stream / raw / обычный).
        5. Вывести результат в stdout.
    """
    args   = parse_args()
    client = OllamaClient(base_url=args.url, model=args.model)

    client.check_server()

    log.info(f"Модель:  {Color.green(args.model)}")
    log.info(f"Промпт:  {Color.green(args.prompt)}")
    log.separator()

    # Потоковый режим
    if args.stream:
        for token in client.generate_stream(args.prompt):
            print(token, end="", flush=True)
        print()  # финальный перенос строки

    # Сырой JSON
    elif args.raw:
        raw = client.generate_raw(args.prompt)
        print(json.dumps(raw, ensure_ascii=False, indent=2))

    # Обычный режим
    else:
        answer = client.generate(args.prompt)
        print(answer)


if __name__ == "__main__":
    main()
