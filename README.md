# RAG-поиск по корпусу научных публикаций arXiv

Система семантического поиска и вопросно-ответного взаимодействия по корпусу научных статей arXiv.
Пользовательский запрос (даже с опечатками) исправляется, по нему из векторного хранилища
извлекаются релевантные фрагменты аннотаций, и LLM формирует ответ **с опорой на найденный контекст**,
а не на общие знания модели.

Реализация: [`rag_arxiv.ipynb`](rag_arxiv.ipynb)

## Стек

| Компонент | Инструмент |
|---|---|
| Источник данных | arXiv API (`export.arxiv.org/api/query`) + `feedparser` |
| Обработка данных | `pandas` |
| Эмбеддинги | `sentence-transformers`, модель `all-MiniLM-L6-v2` |
| Векторное хранилище | `FAISS` (`IndexFlatL2`) |
| LLM | `google/flan-t5-base` (`transformers`, seq2seq) |
| Исправление опечаток | `symspellpy` (SymSpell) |

## Архитектура пайплайна

```
Запрос пользователя
        |
[SymSpell] исправление опечаток
        |
[all-MiniLM-L6-v2] эмбеддинг запроса
        |
[FAISS] поиск top-5 ближайших чанков  <- индекс, построенный из аннотаций arXiv
        |
[Промпт: роль + контекст + вопрос]
        |
[FLAN-T5-base] генерация ответа
```

## Этапы работы

### 1. Извлечение данных из arXiv API

Запросы к arXiv API по тематике `all:machine learning`. Так как API возвращает часть записей
с дубликатами заголовков, `max_results` увеличивается в цикле до тех пор, пока не наберётся
**1000 уникальных публикаций**. Из каждой записи извлекаются `title`, `authors`, `abstract`
и складываются в `pandas.DataFrame`.

### 2. Разбиение на чанки

Аннотации делятся на чанки по **200 слов с перекрытием 30 слов**. Размеры уменьшены
относительно предложенных в задании 500/50 осознанно: все аннотации arXiv короче 500 слов,
поэтому при исходных параметрах разбиение не давало бы эффекта — весь абстракт попадал бы
в один чанк. Итог: **1349 чанков** из 1000 аннотаций.

### 3. Векторизация и построение хранилища

Все чанки кодируются моделью `all-MiniLM-L6-v2` в плотные векторы, из которых строится
FAISS-индекс `IndexFlatL2` (точный поиск по евклидову расстоянию). Размерность индекса
определяется автоматически по форме эмбеддингов.

### 4. Интеграция с LLM и поиск по корпусу

- `corpus_search(query, chunks, top_k=5)` — кодирует запрос и возвращает top-5 ближайших чанков.
- `generate_answer(prompt)` — прямой вызов FLAN-T5 (`max_new_tokens=300`, `top_p=0.9`, `temperature=0.3`).
- `generate_answer_with_RAG(prompt, chunks)` — собирает промпт вида
  `Role: machine learning expert. Use next parts of context to answer the question. Context: {...} Question: {...}`
  и передаёт его в LLM. Назначение роли и явная инструкция «отвечай по контексту» — часть промпт-инжиниринга.

### 5. Обработка опечаток

SymSpell (`max_dictionary_edit_distance=2`, `prefix_length=7`) со словарём частотности
английского языка `frequency_dictionary_en_82_765`. Метод `lookup_compound` исправляет
сразу всю фразу, а не отдельные слова.

## Результаты

Тестирование на 5 запросах, **намеренно содержащих опечатки**, — проверяются сразу два механизма:
исправление запроса и вклад RAG. Ответы сравниваются с ответами той же LLM без контекста.

| Запрос (после исправления) | Без RAG | С RAG |
|---|---|---|
| what is parallelism of machine learning? | `parallel` | `Multivariate Learning.` |
| which benefit of using deep learning for weather? | `a better understanding of the weather` | `We propose a neural network architecture that can predict urban land surface processes using a combination of ML and deep learning.` |
| how machine learning can help to find cancer tumors? | `a machine learning algorithm` | `We compare the accuracy of the ML algorithms on the Wisconsin Diagnostic Breast Cancer dataset by comparing their classification test accuracy and their sensitivity and specificity values.` |
| what is classification model for ecology problems? | `a phylogenetic model` | `Bioclimatic models are a key tool for predicting the range of organisms as a function of climate.` |

Опечатки (`Whot`, `deeeep`, `back propogation`, `machine learnin`, `clasification`) корректно
исправляются SymSpell до подачи в поиск.

**Вывод:** без RAG модель выдаёт короткие тавтологичные ответы, фактически перефразируя вопрос.
С RAG ответы содержательны и опираются на конкретные исследования из корпуса —
названия датасетов, методы, постановки задач. RAG действительно повышает качество ответов LLM.

## Запуск

```bash
pip install feedparser faiss-cpu symspellpy sentence-transformers transformers pandas
```

Ноутбук рассчитан на среду с GPU (модель переносится на `cuda`); для запуска на CPU
нужно убрать вызовы `.to("cuda")`. Ноутбук выполняется последовательно сверху вниз,
внешние ключи и токены не требуются.

## Ограничения и направления развития

- `flan-t5-base` — небольшая модель, длина ответа ограничена; на более крупной модели
  (`flan-t5-large`/`xl`) качество генерации по тому же контексту заметно выше.
- Индекс `IndexFlatL2` даёт точный, но линейный по размеру корпуса поиск. Для масштабирования
  за пределы десятков тысяч чанков стоит перейти на `IndexIVFFlat` или HNSW.
- В индексе хранятся только тексты чанков без привязки к метаданным статьи —
  добавление `title`/`authors` в контекст позволило бы отвечать со ссылкой на источник.
