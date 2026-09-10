# Pre-Entrega 2 — Pipeline de procesamiento validado

Un **pipeline de extracción de entidades técnicas** construido con LangChain y LCEL:
recibe un párrafo de texto libre (un log de error, la descripción de una arquitectura)
y devuelve **siempre** un objeto Pydantic validado — o lanza una excepción después de
agotar reintentos y fallback. Nunca devuelve texto crudo, un `dict` sin validar ni un
valor por defecto silencioso.

```
{"text": "..."} ──▶ prompt | model.with_structured_output(TechnicalEntities) | check
                    └─ .with_retry(backoff exponencial + jitter)
                       └─ reintento consciente del error (feedback de validación)
                          └─ .with_fallbacks([modelo más liviano])
                                                          ──▶ TechnicalEntities
```

Todo el código (identificadores, logs, docstrings) está en inglés. Los nombres que la
consigna escribe en español aparecen traducidos; la tabla de equivalencias está en la
sección [Cómo se cumple cada requisito](#cómo-se-cumple-cada-requisito).

## Instalación

Requiere **Python 3.12**.

```bash
# desde esta carpeta
py -3.12 -m venv .venv          # Windows (Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # (source .venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env            # después completá tu API key
```

Las dependencias están **fijadas a versión exacta** en `requirements.txt` (fecha de
fijación incluida): LangChain cambia rápido y la consigna no fija versiones.

## Configuración

- **`.env`** — sólo secretos, está en `.gitignore`. Alcanza con **una** clave: la del
  proveedor con el que vas a correr. `GOOGLE_API_KEY` tiene free tier en
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey) y es el proveedor por
  defecto. `OPENAI_API_KEY` y `ANTHROPIC_API_KEY` son opcionales.
- **[`config.yaml`](config.yaml)** — todo lo demás: el proveedor por defecto
  (`default_provider`), los **IDs de modelo** por proveedor (`models`), el modelo de
  fallback de cada uno (`fallback_models`), y en `defaults` la temperatura, el timeout
  por llamada, la cantidad de intentos y los parámetros del backoff.

Nada tuneable está hardcodeado. El YAML se parsea al arranque en un modelo Pydantic
validado (`AppSettings`, en `src/settings.py`), así que un archivo mal formado —un
proveedor desconocido, `max_attempts: 0`, un backoff con `max < initial`, un proveedor
sin modelo— **falla rápido y con un error claro** en vez de romper a mitad de un
request. Quitar la entrada de un proveedor en `fallback_models` lo deja sin fallback.
La variable de entorno `EXTRACTION_PIPELINE_CONFIG` apunta a un YAML alternativo (los
tests la usan para probar que el comportamiento sale del archivo).

| Proveedor | Modelo primario | Fallback (más liviano) |
|---|---|---|
| `gemini` (default) | `gemini-3.5-flash` | `gemini-flash-lite-latest` |
| `openai` | `gpt-4o-mini` | `gpt-4.1-mini` |
| `anthropic` | `claude-sonnet-4-6` | `claude-haiku-4-5` |

Los tres modelos se instancian con `temperature=0` (extracción = determinismo, no
creatividad), `timeout=30 s` (ambos de `config.yaml`) y `max_retries=0`: los reintentos internos del SDK se
apagan para que `.with_retry()` sea **la única capa de reintentos**, y por lo tanto
observable en los logs.

## Cómo correr

```bash
python -m src.main                    # proveedor por defecto: gemini
python -m src.main --provider openai  # o anthropic
python -m src.main --text "..."       # un texto propio en lugar de los dos casos
python -m src.main --log-level WARNING
```

`src/main.py` es el **mini-script de prueba asíncrono** de la consigna: ejecuta con
`asyncio.run()` dos casos a través de `process_text()` — un texto técnico claro y la
prueba de estrés con un texto ambiguo — e imprime cada resultado con
`model_dump_json(indent=2)`. Cada caso está aislado en su propio `try/except`, así una
falla no oculta la otra; el código de salida es 1 si alguno falló.

Salida real (`python -m src.main`, 2026-09-07, Gemini):

```
INFO    extraction_pipeline: [gemini] Processing text (242 characters)
INFO    httpx: HTTP Request: POST .../models/gemini-3.5-flash:generateContent "HTTP/1.1 200 OK"
INFO    extraction_pipeline: Model responded: finish_reason=STOP input_tokens=119 output_tokens=581
INFO    extraction_pipeline: [gemini] Validated extraction: {'technologies': ['FastAPI', 'Redis', 'PostgreSQL'], 'criticality_level': 'high', ...}

=== Clear technical text ===
Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en Redis parece
saturarse en picos de tráfico y las conexiones a PostgreSQL se agotan porque el pool está
mal dimensionado. Esto está afectando a usuarios en producción.

{
  "technologies": ["FastAPI", "Redis", "PostgreSQL"],
  "criticality_level": "high",
  "technical_summary": "La API desarrollada con FastAPI presenta timeouts intermitentes en producción debido a la saturación del caché en Redis y al agotamiento de las conexiones en el pool de PostgreSQL."
}
```

Sin clave configurada el script falla **con un mensaje, no con un traceback**:

```
FAILED: ValueError: OPENAI_API_KEY is not set; add it to .env (see .env.example)
```

## Cómo se cumple cada requisito

Equivalencia entre los nombres de la consigna y los del código:

| Consigna (español) | Código (inglés) | Dónde |
|---|---|---|
| `EntidadesTecnicas` | `TechnicalEntities` | `src/schemas.py` |
| `NivelCriticidad` · `baja` / `media` / `alta` | `CriticalityLevel` · `low` / `medium` / `high` | `src/schemas.py` |
| `tecnologias` | `technologies` | `src/schemas.py` |
| `nivel_de_criticidad` | `criticality_level` | `src/schemas.py` |
| `resumen_tecnico` | `technical_summary` | `src/schemas.py` |
| variable `{texto}` del prompt | `{text}` (clave de invocación `"text"`) | `src/chain.py` |
| `schemas.py`, `chain.py`, `process_text()` | mismos nombres, dentro de `src/` | `src/` |

### Banda 1 — Definición del Esquema y Salida Estructurada (30 %)

- `src/schemas.py`: `TechnicalEntities` con exactamente los tres campos pedidos —
  `technologies: List[str]` (`min_length=1`), `criticality_level: CriticalityLevel`
  (enum cerrado de tres valores, `class CriticalityLevel(str, Enum)`) y
  `technical_summary: str` (`min_length=10`). Cada campo lleva `description=`: con
  `with_structured_output` esas descripciones son lo que el proveedor le muestra al
  modelo, es decir, son *prompt engineering embebido en el contrato*.
- `@field_validator("technologies")`: la capa **semántica**. `min_length=1` aceptaría
  `["  ", ""]`; el validador limpia espacios, descarta vacíos, deduplica preservando el
  orden y lanza `ValueError` si no queda nada.
- `src/chain.py`, `_resilient_extraction()`:
  `prompt | model.with_structured_output(TechnicalEntities, include_raw=True) | check`.
  La cadena devuelve una **instancia** de `TechnicalEntities`, nunca un `AIMessage`.
- Evidencia: `tests/test_schemas.py` (10 tests) prueba lo que el esquema rechaza —
  lista vacía, entradas en blanco, `"HIGH"`/`"alta"` fuera del enum, resumen corto,
  campo faltante — y `test_happy_path_returns_a_validated_instance` verifica el tipo
  de retorno de la cadena.

### Banda 2 — Implementación de Cadena LCEL y Prompting (30 %)

- El flujo principal está compuesto con el operador `|` (`src/chain.py`,
  `_resilient_extraction()`): `prompt | structured_model | RunnableLambda(check)`.
  No hay ninguna secuencia imperativa "llamar y parsear".
- `prompt` es un `ChatPromptTemplate.from_messages` con roles separados: `("system", …)`
  fija el rol (analista técnico), el objetivo y la descomposición de la tarea;
  `("human", "{text}")` lleva **sólo** el dato. Un `MessagesPlaceholder("feedback",
  optional=True)` al final es el gancho del reintento consciente del error; las
  invocaciones normales pasan sólo `{"text": ...}`.
- No hay f-strings: `tests/test_prompt.py` verifica que la variable de entrada es
  exactamente `{"text"}`, que hay un `SystemMessage` y un `HumanMessage`, y que las
  llaves dentro del texto del usuario sobreviven (sustitución, no interpolación).

### Banda 3 — Ejecución Asíncrona y Resiliencia (25 %)

- `process_text(text, provider)` es `async` y ejecuta `await chain.ainvoke({"text": text})`
  (`src/chain.py`). Loguea a INFO antes (proveedor, longitud), a INFO el `model_dump()`
  validado en el éxito, y a ERROR **re-lanzando** cuando todo falló — incluidas las
  fallas al construir la cadena (clave faltante, config inválida). No existe un tercer
  resultado. La cadena también funciona con `invoke()` sincrónico: el paso de reintento
  consciente define ambos cuerpos.
- `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True, ...)` envuelve la
  cadena compuesta. Además de los errores transitorios cubre el **JSON incompleto**: el
  paso `check` lee el `finish_reason` de la respuesta cruda (`include_raw=True`) y lanza
  `TruncatedResponseError` si el proveedor cortó por tokens (`length` / `max_tokens` /
  `MAX_TOKENS`) — es el "error a evitar" de la consigna, detectado explícitamente en
  lugar de sólo rebotar contra la validación.
- Los logs permiten observar el flujo completo: cada intento fallido
  (`Model call failed with …`), cada reintento (`Retry attempt 2/3 …`), el cambio de
  modelo (`switching to the fallback model`), el `finish_reason` y los tokens de cada
  respuesta, y el reintento con feedback.
- Evidencia: `tests/test_chain.py` (17 tests) cubre reintento ante 429/timeout,
  truncamiento, reintento con feedback, fallback, no-reintento de un 401, y que
  `process_text` loguea y re-lanza.

### Banda 4 — Estructura del Repositorio y Script de Prueba (15 %)

```
pre-entrega-2/
├── src/
│   ├── schemas.py      # contrato de datos: CriticalityLevel, TechnicalEntities
│   ├── chain.py        # get_model(), prompt, compose_chain(), build_chain(), process_text()
│   ├── settings.py     # config.yaml -> AppSettings (Pydantic); claves desde .env
│   └── main.py         # mini-script asíncrono: asyncio.run() sobre ambos casos
├── config.yaml         # proveedor por defecto, modelos, fallback, timeout, reintentos
├── requirements.txt    # versiones exactas
├── pytest.ini
├── .env.example        # placeholders vacíos; .env está en .gitignore
├── .gitignore
├── README.md · README.en.md
└── tests/              # 35 tests, todos offline
    ├── conftest.py     # ScriptedChatModel (modelo falso con guion) + config.yaml temporal
    ├── test_schemas.py
    ├── test_prompt.py
    ├── test_chain.py
    └── test_settings.py
```

## Resiliencia: qué falla y quién lo atiende

La taxonomía de errores de la clase (transitorios / de formato / permanentes) se mapea
uno a uno a las capas de la cadena. Las clases de excepción son las **normalizadas de
`langchain-core`** (`ModelRateLimitError`, `ModelTimeoutError`, …): las tres
integraciones de proveedor traducen sus errores de SDK a esas clases, así que la
política es independiente del proveedor y no importa ningún SDK.

| Falla | Ejemplo | Capa que la atiende | Qué hace |
|---|---|---|---|
| **Transitoria** | 429, timeout, conexión, 5xx, respuesta truncada | `.with_retry()` | hasta 3 intentos, espera `min(1·2ⁿ + jitter, 10) s`. Incluye los `httpx.TimeoutException` / `ConnectError` crudos que el SDK de Gemini no envuelve |
| **De formato** | el objeto no valida (`ValidationError`, `OutputParserException`, refusal de OpenAI, `ValueError` del parser) | reintento consciente del error | todas se normalizan en `SchemaValidationError`; **una** re-invocación con la respuesta rechazada como turno del asistente y el texto del error de Pydantic como mensaje adicional: el modelo ve *qué* dijo y *qué* campo falló |
| **Modelo caído** | el primario agotó sus capas | `.with_fallbacks()` | la misma pila completa sobre un modelo más liviano del mismo proveedor |
| **Permanente** | 401, 400, modelo inexistente | ninguna | un intento, fallback, y la excepción llega al llamador con su mensaje |

El reintento consciente es distinto del reintento ciego, y va *adentro* del fallback a
propósito: ante un error de validación primero se le da al modelo la chance de
corregirse (slide 38 de la clase); recién si el modelo falla repetidamente se cambia de
modelo (slide 22).

Esto no es teórico. En la primera corrida real del 2026-09-07 el modelo primario de
Gemini estaba saturado y la pila entera se ejerció sola:

```
WARNING extraction_pipeline: Model call failed with GoogleAPIError: 503 UNAVAILABLE. {... 'This model is currently experiencing high demand ...'}
WARNING extraction_pipeline: Retry attempt 2/3 after exponential backoff
WARNING extraction_pipeline: Model call failed with GoogleAPIError: 503 UNAVAILABLE. ...
WARNING extraction_pipeline: Retry attempt 3/3 after exponential backoff
WARNING extraction_pipeline: Model call failed with GoogleAPIError: 503 UNAVAILABLE. ...
WARNING extraction_pipeline: Primary model failed with GoogleAPIError: 503 UNAVAILABLE. ...; switching to the fallback model
INFO    httpx: HTTP Request: POST .../models/gemini-flash-lite-latest:generateContent "HTTP/1.1 200 OK"
INFO    extraction_pipeline: Model responded: finish_reason=STOP input_tokens=119 output_tokens=84
INFO    extraction_pipeline: [gemini] Validated extraction: {'technologies': ['FastAPI', 'Redis', 'PostgreSQL'], 'criticality_level': 'high', ...}
```

Tres intentos con backoff, cambio al fallback, objeto validado. El llamador nunca se
enteró.

## Prueba de estrés: texto ambiguo

Entrada: *"El sistema anda medio raro últimamente, no sé bien qué está pasando."* — no
menciona ninguna tecnología.

**¿Lanza el validador o se recupera el modelo?** Se recupera el modelo, y lo hace
inventando. En tres corridas reales (dos con el fallback `gemini-flash-lite-latest`
durante la saturación, una con `gemini-3.5-flash`):

| Corrida | `technologies` | `criticality_level` |
|---|---|---|
| 1 | `["sistema"]` | `low` |
| 2 | `["Sistema"]` | `medium` |
| 3 | `["Sistema"]` | `low` |

Observaciones:

- El esquema exige `min_length=1` en `technologies`, y el modelo lo satisface con un
  **placeholder** (`"sistema"`, la única palabra con aspecto técnico del texto). El
  contrato se cumple *a la letra* y se esquiva *en espíritu*: la validación estructural
  garantiza **forma, no verdad**. Es el "piso, no el techo" de la clase.
- La criticidad no es estable entre corridas (`low` / `medium`) aun con
  `temperature=0`: el texto no da información para decidirla, así que el modelo elige.
- El resumen es honesto en los tres casos ("sin especificar componentes, herramientas o
  errores concretos"): el campo de texto libre refleja la ambigüedad mejor que la lista
  cerrada.

Lo que haría falta para que este caso *falle* en lugar de pasar es una regla semántica
cruzada (un `model_validator` que rechace placeholders o exija tecnologías reales cuando
la criticidad es alta). Está fuera del alcance de la consigna y se deja documentado como
límite conocido.

## Tests

```bash
python -m pytest -v
```

35 tests, todos **offline**: `tests/conftest.py` define `ScriptedChatModel`, un
`BaseChatModel` falso que reproduce un guion (un `AIMessage` con tool call → se
devuelve; una excepción → se lanza) y registra cada lista de mensajes que recibió. Así
se prueba la cadena real, con `with_structured_output` real, sin red ni claves.

- `test_schemas.py` (10) — lo que el contrato acepta y rechaza; el validador semántico.
- `test_prompt.py` (4) — variable exacta `text`, roles, placeholder de feedback, no-f-string.
- `test_settings.py` (8) — el `config.yaml` entregado es válido; un YAML alternativo
  (vía `EXTRACTION_PIPELINE_CONFIG`) cambia proveedor por defecto y modelos; el fallback
  es opcional por proveedor; un archivo mal formado falla al arrancar con
  `ValidationError` (proveedor desconocido, `max_attempts: 0`, temperatura fuera de
  rango, backoff incoherente, proveedor sin modelo).
- `test_chain.py` (17) — instancia validada en el camino feliz (async y sync); 429 +
  timeout reintentados con backoff y logueados; `httpx.TimeoutException` crudo (Gemini)
  reintentado; `finish_reason=length` detectado y reintentado; error de validación →
  segunda llamada que contiene la respuesta anterior del modelo y el mensaje de Pydantic
  con el nombre del campo; respuesta sin salida estructurada normalizada y reintentada
  con feedback; segunda falla de validación se propaga; fallback tras agotar reintentos
  (primario llamado exactamente 3 veces); 401 **no** reintentado (1 llamada);
  `process_text` loguea INFO y devuelve / loguea ERROR y re-lanza, también cuando la
  cadena no se puede construir; proveedor
  desconocido y clave faltante fallan con mensaje claro; el ID de modelo, el timeout y la
  temperatura del modelo salen de `config.yaml` y `max_retries=0`; `build_chain()` sin
  argumento usa el `default_provider` y el fallback del archivo.

## Decisiones y alcance

- **Proveedor por defecto: Gemini.** La consigna nombra `ChatOpenAI` o `ChatAnthropic`;
  los tres están implementados y son intercambiables con `--provider`, pero el default
  es el que tiene free tier. Los IDs de Gemini se verificaron contra la API el
  2026-09-07: `gemini-flash-latest` devolvía 503 sostenido, `gemini-3.6-flash` (el del
  notebook de la cursada) funciona pero **ignora `temperature`** con un warning,
  `gemini-2.5-flash` ya no está disponible; `gemini-3.5-flash` respeta `temperature=0`
  y responde en ~4 s. Los IDs de OpenAI/Anthropic no se pudieron ejercer sin clave.
- **`langchain` (metapaquete) no está en `requirements.txt`**: nada lo importa. Todo
  viene de `langchain-core` y de los tres paquetes de integración.
- **No implementado, a propósito** (fuera del alcance de la consigna): `model_validator`
  cruzado, LangSmith, cache, logging JSON, métricas de TTFT/costo.
- **Sin clave para OpenAI/Anthropic** las rutas están cubiertas por los tests offline y
  por la verificación de tipos de excepción normalizados; no por una corrida real.

---

An English version of this document is available in [README.en.md](README.en.md).
