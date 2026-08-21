# Pre-Entrega 1 — Cliente de LLM robusto y asíncrono

Un cliente de LLM **agnóstico al proveedor, asíncrono y resiliente**: las mismas
dos llamadas (`generate` / `generate_stream`) funcionan contra OpenAI, Anthropic
o Gemini, y el manager sobrevive a rate limits, timeouts y caídas de proveedor
reintentando con backoff exponencial y cambiando de proveedor automáticamente.

## Instalación

Requiere **Python 3.12**.

```bash
# desde esta carpeta
py -3.12 -m venv .venv          # Windows (en Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # (source .venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env            # después completá tus API keys
```

## Configuración

Dos fuentes de configuración, deliberadamente separadas:

- **`.env`** — **sólo secretos** (está en `.gitignore`). Alcanza con **una sola
  clave**: `GOOGLE_API_KEY` tiene free tier en
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey) y no pide
  tarjeta.
- **[`config.yaml`](config.yaml)** — todo lo demás: temperatura por defecto,
  tope de tokens, política de reintentos, timeouts, tamaño del semáforo, los
  **IDs de modelo** y el orden de fallback.

Nada tuneable está hardcodeado. El YAML se parsea al arranque en un modelo
Pydantic validado (`AppSettings`, en `src/settings.py`), así que un archivo mal
formado **falla rápido y con un error claro** en vez de romper a mitad de un
request. Incluso los *límites de validación* (por ejemplo, el rango de
temperatura aceptado) salen del archivo de configuración: cambiar un número en
el YAML cambia qué se acepta, sin tocar una línea de código.

## Cómo correr los ejemplos

Todo lo ejecutable vive en **un único punto de entrada, `src/main.py`**, y se
elige con un modo posicional. Se corre como módulo, desde esta carpeta:

```bash
python -m src.main normal      # una llamada completa; respuesta + latencia total
python -m src.main streaming   # token a token + TTFT / total / throughput
python -m src.main fallback    # un primario que falla -> failover automático
python -m src.main gather      # 5 prompts en secuencia, después con asyncio.gather
python -m src.main all         # los cuatro anteriores, en orden
```

| Modo | Qué demuestra |
|---|---|
| `normal` | `await manager.generate(...)`: la respuesta llega entera de una vez. |
| `streaming` | `async for chunk in manager.generate_stream(...)`: el primer token llega en ~TTFT en vez de en ~latencia total. |
| `fallback` | El primario no puede responder y el manager cambia de proveedor solo. |
| `gather` | Las mismas 5 llamadas secuenciales y después con `asyncio.gather`. |
| `all` | Los cuatro seguidos, con un encabezado por sección. |

Opciones comunes: `--prompt TEXTO`, `--max-tokens N`, `--quiet` (silencia el log
estructurado, útil para capturas limpias) y `--simulate {bad-key,rate-limit}`
para el modo `fallback`. `python -m src.main --help` las lista todas.

### El modo `fallback` funciona con una sola API key

- **`--simulate bad-key`** (default) rompe a propósito la clave del primario y
  vuelve a insertar el primario *sano* como primer fallback. Así el failover se
  puede demostrar aunque haya un único proveedor configurado. Una clave
  inválida es un error **no reintentable** (401 en OpenAI/Anthropic, 400
  `API_KEY_INVALID` en Gemini), así que el cambio es inmediato.
- **`--simulate rate-limit`** pone al frente de la cadena un cliente falso que
  siempre devuelve 429. Rate limit **sí** es reintentable, así que este modo
  ejercita el camino completo de resiliencia — `max_attempts` intentos con
  backoff exponencial y jitter — *antes* de que el manager se rinda y cambie de
  proveedor. No gasta créditos ni necesita red, y es literalmente el paso 5 de
  la consigna ("probá tu cliente simulando un error de rate limit").

## Arquitectura

```
lógica de negocio ──▶ AsyncLLMManager ──▶ BaseLLMClient (ABC)
                      factory · semáforo      ├── OpenAIClient
                      timeout · retry         ├── AnthropicClient
                      fallback · métricas     └── GeminiClient

src/main.py = sólo presentación: imprime, mide y simula fallas.
```

`src/main.py` no tiene ninguna política propia: cada decisión (retry, fallback,
timeout, semáforo) vive en `AsyncLLMManager`. Loguru se configura ahí y no
dentro de la librería a propósito — una librería no debería instalar efectos
secundarios de logging al ser importada.

- **`BaseLLMClient` (ABC)** — el contrato: `generate()` devolviendo un
  `ModelResponse` validado, y `generate_stream()` como generador asíncrono. Una
  subclase que se olvide de implementar un método falla al instanciarse, no en
  el medio de un request.
- **Factory** — `AsyncLLMManager._create_client()` es el único lugar que mapea
  `Provider -> clase concreta`. Cambiar de proveedor es un cambio de
  configuración (**interoperabilidad**), y la lógica de negocio sólo sostiene la
  abstracción, nunca un SDK.
- **Qué absorbe la abstracción** — los tres SDKs no se ponen de acuerdo en casi
  nada: nombres de método, dónde vive el texto de la respuesta, la forma de
  streamear y cómo se pasa el rol `system` (OpenAI: dentro de `messages`;
  Anthropic: un parámetro `system=` de primer nivel; Gemini:
  `system_instruction` en la config). Cada cliente normaliza el dialecto de su
  proveedor detrás de la misma interfaz.
- **Objetos-resultado, no excepciones** — los clientes nunca dejan escapar una
  excepción del proveedor: toda falla se vuelve
  `ModelResponse(error=..., error_kind=...)`. Ese `error_kind` tipado es lo que
  hace posible la capa de resiliencia sin parsear strings. Los errores *de
  configuración* irrecuperables (falta la clave, proveedor no soportado) hacen
  lo contrario y **rompen al arranque**: fallar claro es mejor que fallar en
  silencio. Los bugs de nuestro propio código se propagan: sólo se captura la
  familia de excepciones tipadas de cada SDK, nunca un `Exception` pelado.

## Resiliencia

- **Política de reintentos** — sólo se reintentan fallas transitorias: rate
  limit (429), errores de conexión, timeouts y 5xx. Auth (401) y bad request
  (400) fallan igual en cada intento, así que vuelven de inmediato (reintentar
  un 401 cinco veces son cinco fracasos idénticos).
- **Fórmula de backoff** —
  `base_delay_s * 2^intento * uniform(jitter_min, jitter_max)`, con los cuatro
  números saliendo de `config.yaml`. El **jitter** importa: sin él, todos los
  clientes que fallaron en el mismo momento reintentan en los mismos momentos
  también, golpeando en oleadas sincronizadas a un servicio que ya está mal.
- **Fallback** — el manager recorre la cadena de proveedores del
  `fallback_order` de `config.yaml`; cuando uno agota sus intentos (o falla con
  un error no reintentable), loguea el cambio y prueba el siguiente. Si fallan
  todos, el `ModelResponse.error` devuelto agrega el detalle de cada intento.
- **Decisión de diseño: fallback en streaming** — el manager cambia de
  proveedor **sólo antes del primer token**. Una vez que hay tokens en pantalla,
  reiniciar en otro proveedor duplicaría la salida, así que una falla a mitad de
  stream se expone dentro del stream en lugar de reintentarse. Los clientes
  señalan una falla previa al primer token con un `StreamError` tipado, para que
  el manager pueda distinguir los dos casos.
- **Timeouts** — en dos capas: `asyncio.timeout(timeout_s)` por llamada en el
  manager (en streaming acota la espera del *primer* token), más el mismo valor
  pasado al `timeout` a nivel cliente de cada SDK.
- **Semáforo** — `asyncio.Semaphore(max_concurrent)` limita los requests en
  vuelo (backpressure / throttling: los demás esperan *dentro* del proceso,
  donde esperar es gratis, en vez de convertirse en un muro de 429s).
- **Logging estructurado** — Loguru registra qué proveedor respondió y en cuánto
  tiempo, cada reintento con su demora, cada fallback con su motivo y cada error
  con su tipo. Las claves son `SecretStr` y nunca se loguean.
- **No implementado** (fuera de alcance, próximo paso natural): un **circuit
  breaker** — dejar de llamar por un rato a un proveedor que falla repetidamente,
  en vez de reintentarlo en cada request.

## Métricas medidas

Corrida real del 2026-08-20, `python -m src.main all` (Gemini free tier,
`gemini-flash-lite-latest`, `max_tokens=200`):

| Métrica | Valor |
|---|---|
| Latencia total, sin streaming (`normal`) | **1678 ms** |
| TTFT (streaming, proveedor primario) | **1438 ms** |
| Latencia total (streaming, mismo pedido) | **1593 ms** |
| Throughput | **19,3 chunks/s** (proxy de tokens/s: cada chunk trae uno o pocos tokens) |
| 5 prompts en secuencia | **4,36 s** |
| 5 prompts con `asyncio.gather` | **1,52 s** — **2,9x** más rápido, las mismas 5 llamadas |
| Failover, `--simulate bad-key` (punta a punta) | **2606 ms** — un round trip rechazado (~1,2 s) más el reintento exitoso; el *cambio en sí* es inmediato, porque una clave inválida no es reintentable |
| Failover, `--simulate rate-limit` (punta a punta) | **6558 ms** — 3 intentos con demoras de **1,11 s** y **2,90 s** (`base 1,0 × 2^n × jitter`), y recién ahí el cambio |

El número de `gather` es la evidencia más legible de que la arquitectura es
genuinamente asíncrona: las mismas cinco llamadas, la misma espera total, pero
**superpuesta** en vez de encolada. El speedup está acotado por
`max_concurrent: 5` y por el rate limiting del propio proveedor, no por el
cliente.

## Tests

```bash
python -m pytest tests/ -v
```

9 tests, todos **offline** (fakes que implementan la ABC — sin gastar créditos y
sin red), en `tests/test_resilience.py`:

- `TestWithRetry` — un 429 reintentado hasta que sale bien, **sin** reintento
  ante un 401 o un 400 (exactamente 1 llamada), el corte en `max_attempts`, y la
  demora de backoff dentro de sus cotas de jitter.
- `TestFallback` — un primario rate-limiteado que recién cede tras agotar sus
  reintentos, un primario con error no reintentable que cede al primer intento,
  una cadena donde fallan todos y se agregan los errores de cada proveedor, y
  una respuesta exitosa que reporta su latencia total.

El test de fallback es el de mayor valor de la suite: cae de lleno en la banda
del 40% (resiliencia). 

---

An English version of this document is available in [README.en.md](README.en.md).
