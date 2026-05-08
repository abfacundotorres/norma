// Norma — Worker v8
// Cambios sobre v7:
// 1. Fix crítico: flujo licencias usa SOLO normativa del RAG — no alucina artículos del Estatuto
// 2. Temperature bajada a 0.35 en flujo general (reducir alucinaciones normativas)
// 3. topK 3 + score 0.70 para licencias (evitar mezcla de chunks)
// 4. Query enriquecido para RAG en flujos wizard (no embeddea "1" o "sí")
// 5. HISTORY_LIMIT bajado a 8 (llama-3.1-8b-instant tiene ventana corta)
// 6. Logging de fallos RAG para detectar degradación silenciosa
// 7. Instrucción explícita de fallback honesto cuando no hay normativa en contexto
// 8. Template de nota formal de licencias corregido — no hardcodea Estatuto Docente
// 9. Métricas enriquecidas: chip_usado, problema_detectado

const GROQ_URL      = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL    = 'llama-3.1-8b-instant';
const GEMINI_URL    = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent';
const GEMINI_EMBED  = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent';
const HISTORY_LIMIT = 8;

const VALID_FLOWS = new Set(['general','actas','comunicados','proyectos','licencias','resumen']);

function isDocumentText(t) {
  return t.includes('[TITULO]') || t.includes('[FIN_DOCUMENTO]');
}

function safe(s) {
  if (!s || typeof s !== 'string') return '';
  return s.replace(/`/g,"'").replace(/\$/g,'').trim().slice(0,80);
}

function getLLMParams(flow) {
  if (flow === 'general')  return { maxTokens: 1600, temperature: 0.35 };
  if (flow === 'resumen')  return { maxTokens: 1000, temperature: 0.3  };
  if (flow === 'licencias') return { maxTokens: 1800, temperature: 0.3  };
  return { maxTokens: 2200, temperature: 0.5 };
}

// ── Métricas ─────────────────────────────────────────────────────

function logEvent(ctx, env, data) {
  if (!env.METRICS_WEBHOOK) return;
  ctx.waitUntil(
    fetch(env.METRICS_WEBHOOK, {
      method: 'POST',
      redirect: 'follow',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tipo:              'evento',
        timestamp:         new Date().toISOString(),
        flow:              data.flow            || 'general',
        municipio:         data.municipio       || 'demo',
        rag_usado:         data.rag_usado       || false,
        rag_chunks:        data.rag_chunks      || 0,
        rag_error:         data.rag_error       || false,
        model_used:        data.model_used      || 'unknown',
        chip_usado:        data.chip_usado       || '',
        problema_detectado: data.problema_detectado || '',
        session_id:        data.session_id        || '',
      }),
    })
    .then(r => console.log('[metrics]', r.status))
    .catch(e => console.error('[metrics error]', e.message))
  );
}

// ── RAG ─────────────────────────────────────────────────────────

// Enriquece el query para flujos wizard donde el mensaje puede ser "1", "sí", etc.
function enrichQuery(message, flow) {
  const m = message.toLowerCase().trim();
  // Si el mensaje es muy corto o es solo un número/confirmación, usar query genérico del flujo
  if (m.length < 15 || /^[1-6sn]$/.test(m) || m === 'sí' || m === 'si' || m === 'no') {
    const flowQueries = {
      licencias: 'licencia docente enfermedad maternidad fallecimiento matrimonio Ley 4356 Córdoba',
      general:   'normativa educativa Córdoba directivo primaria',
      actas:     'acta reunión escuela primaria Córdoba',
    };
    return flowQueries[flow] || message;
  }
  // Para licencias, enriquecer con contexto normativo
  if (flow === 'licencias') {
    return `licencia docente ${message} Ley 4356 Córdoba normativa`;
  }
  return message;
}

async function getEmbedding(text, apiKey) {
  try {
    const res = await fetch(`${GEMINI_EMBED}?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'models/gemini-embedding-001',
        content: { parts: [{ text: text.slice(0, 1000) }] },
        taskType: 'RETRIEVAL_QUERY',
      }),
    });
    if (!res.ok) {
      console.error('[RAG] Embedding error:', res.status);
      return null;
    }
    const data = await res.json();
    return data.embedding?.values?.slice(0, 1536) || null;
  } catch(e) {
    console.error('[RAG] Embedding exception:', e.message);
    return null;
  }
}

async function retrieveContext(query, env, flow) {
  if (!env.VECTORIZE || !env.GEMINI_API_KEY) return { context: '', count: 0, sources: [], error: false };
  try {
    const enriched = enrichQuery(query, flow);
    const vector = await getEmbedding(enriched, env.GEMINI_API_KEY);
    if (!vector) return { context: '', count: 0, sources: [], error: true };

    // Para licencias: umbral más alto para evitar mezcla de chunks irrelevantes
    const topK      = flow === 'licencias' ? 3 : 4;
    const minScore  = flow === 'licencias' ? 0.70 : 0.65;

    const results = await env.VECTORIZE.query(vector, { topK, returnMetadata: true });
    if (!results?.matches?.length) return { context: '', count: 0, sources: [], error: false };

    const validMatches = results.matches.filter(m => m.score > minScore);
    if (!validMatches.length) return { context: '', count: 0, sources: [], error: false };

    const chunks = [], sources = [];
    for (const m of validMatches) {
      const text   = m.metadata?.text    || '';
      const fuente = m.metadata?.articulo || m.metadata?.area || m.metadata?.fuente || '';
      if (!text) continue;
      chunks.push(text);
      if (fuente && !sources.includes(fuente)) sources.push(fuente);
    }
    if (!chunks.length) return { context: '', count: 0, sources: [], error: false };

    const sourceLine = sources.length
      ? `\nFuentes recuperadas: ${sources.join(' | ')}`
      : '';

    return {
      context: `\n\nNORMATIVA RELEVANTE (usá SOLO esta normativa — no cites artículos de memoria):${sourceLine}\n` + chunks.join('\n---\n'),
      count: chunks.length,
      sources,
      error: false,
    };
  } catch(e) {
    console.error('[RAG] Query exception:', e.message);
    return { context: '', count: 0, sources: [], error: true };
  }
}

// ── Prompt ──────────────────────────────────────────────────────

function buildPrompt(flow, docenteCtx = {}, ragContext = '') {
  const now     = new Date();
  const fecha   = now.toLocaleDateString('es-AR',{day:'numeric',month:'long',year:'numeric'});
  const mes     = now.getMonth()+1;
  const momento = mes<=3?'inicio de año':mes<=6?'primer semestre':mes<=9?'segundo semestre':'cierre de año';

  const nombre = safe(docenteCtx.nombre);
  const cargo  = safe(docenteCtx.cargo);
  const ctxLine = (nombre||cargo)
    ? '\nDatos ya conocidos: '+(nombre?`Nombre: ${nombre}.`:'')+' '+(cargo?`Cargo: ${cargo}.`:'')+' Usá estos datos en los documentos sin volver a preguntar.'
    : '';
  const campoDirectivo = nombre
    ? `[CAMPO] Directivo/a: ${nombre}${cargo?' ('+cargo+')':''}`
    : '[CAMPO] Directivo/a:';

  const base =
`Sos Norma, directora con 25 años de experiencia en primaria cordobesa. ${fecha} (${momento}).${ctxLine}
Hablás con directivos, vicedirectores y equipos de conducción de primaria en Córdoba, Argentina.
Tono: profesional, cálido, directo. Usás "vos". 1-2 emojis máximo.
Ejemplos de tono: "Dale, te armo el acta ahora." / "Bárbaro, ya entendí." / "Necesito un dato más..."
NUNCA: "Entendido" / "Por supuesto" / recapitular antes de generar.

NORMATIVA — REGLA CRÍTICA:
Cuando citás normativa, usá EXCLUSIVAMENTE la que aparece en "NORMATIVA RELEVANTE" del contexto.
Si no hay normativa en el contexto o no encontrás el artículo exacto, decí claramente: "No tengo el artículo exacto — verificá con supervisión o en el texto completo de la norma."
NUNCA inventes números de artículos ni cites normativa de memoria para licencias.
Cuando sí tenés la fuente, citá brevemente entre paréntesis: (Ley 4356, art. X) / (Ley 9905) / etc.

FORMATO DOCUMENTOS — obligatorio:
Marcas (una por línea): [TITULO] [SECCION] [SUBSECCION] [CAMPO] [TEXTO] [LISTA] [FIN_DOCUMENTO] [SEPARADOR]
[LISTA] siempre con texto en la misma línea: "[LISTA] ítem"
NUNCA guiones ni asteriscos como listas. Dato no provisto → "(completar)".${ragContext}`;

  switch (flow) {

    case 'licencias': return base + `

FLUJO: LICENCIAS DOCENTES
Ayudás con consultas y gestión de licencias usando EXCLUSIVAMENTE la normativa del contexto RAG.
La normativa principal de licencias docentes en Córdoba es la Ley 4356, Ley 9905 y Ley 10318 — NO el Estatuto Docente.
Si no hay normativa en el contexto para una consulta específica, decilo claramente.
Hacé UNA pregunta por turno. Esperá cada respuesta.

P1: "¿Qué tipo de licencia necesitás gestionar?
1. Enfermedad (propia o de familiar a cargo)
2. Maternidad / paternidad / adopción
3. Examen o estudio
4. Duelo (fallecimiento de familiar)
5. Matrimonio
6. Violencia familiar o de género
7. Otra — describí"
[Esperá respuesta]
P2: "¿El docente ya inició la licencia o todavía no? ¿Tenés el certificado médico o documentación requerida?"
[Esperá respuesta]
P3: "¿Cuántos días lleva o se estima que durará?"
[Esperá respuesta]
P4: "¿Necesitás solo orientación normativa, o también una nota formal para el legajo / supervisión?"
[Si pide nota → generá con el formato abajo. Si solo orientación → respondé con pasos y normativa del contexto RAG.]

NOTA FORMAL DE LICENCIA:
[TITULO] Nota de Licencia — {tipo} — {apellido docente} — {fecha}
[SECCION] Datos del docente
[CAMPO] Apellido y nombre:
[CAMPO] Cargo:
[CAMPO] Sección/Grado:
[CAMPO] Fecha de inicio de licencia:
[CAMPO] Duración estimada:
[CAMPO] Documentación presentada:
${campoDirectivo}
[SECCION] Fundamento normativo
[TEXTO] {citar SOLO el artículo que aparece en la NORMATIVA RELEVANTE del contexto — si no tenés el artículo exacto, escribir "(verificar artículo específico con supervisión)"}
[SECCION] Observaciones
[TEXTO] (completar)
[SECCION] Resolución
[TEXTO] En virtud de lo expuesto y de acuerdo con la normativa vigente, se otorga la licencia solicitada a partir de la fecha indicada, quedando constancia en el legajo personal del/la docente.
[CAMPO] Firma y sello directivo/a:
[CAMPO] Fecha:
[FIN_DOCUMENTO]

Después: "¿Necesitás algo más sobre esta licencia o gestionar otra?"`;

    case 'resumen': return base + `

FLUJO: RESUMEN DE CONVERSACIÓN
Generás un resumen ejecutivo de la conversación en formato checklist descargable.
No hacés preguntas — procesás el historial directamente y generás el documento.

[TITULO] Resumen de consulta — Norma · Aula Clara — {fecha}
[SECCION] Tema tratado
[TEXTO] {tema principal de la conversación en 1-2 oraciones}
[SECCION] Normativa consultada
[LISTA] {norma o artículo mencionado — uno por línea. Si no hubo normativa, escribir "Sin normativa específica citada"}
[SECCION] Pasos acordados / Acciones a tomar
[LISTA] {acción concreta — una por línea, ordenadas por prioridad. Verbo en infinitivo: "Convocar al docente", "Registrar en acta", etc.}
[SECCION] Documentos generados en esta sesión
[LISTA] {título del documento si se generó alguno, o "Ninguno"}
[SECCION] Observaciones
[TEXTO] {cualquier advertencia, plazo o dato importante que el directivo deba recordar}
[FIN_DOCUMENTO]`;

    case 'actas': return base + `

FLUJO: ACTA DE REUNIÓN
Documento legal — debe ser preciso y formal.
Hacé UNA pregunta por turno. Esperá cada respuesta antes de continuar.
SIEMPRE empezá con P1 — aunque el usuario ya haya mencionado el tipo de reunión.

P1: "¿Qué tipo de reunión fue?
1. Reunión de personal docente
2. Reunión con familias (general o por grado)
3. Reunión con supervisión
4. Otra — describí"
[Esperá respuesta]
P2: "¿Cuándo fue? (fecha y hora de inicio)"
[Esperá respuesta]
P3: "¿Quiénes participaron? (cargos y cantidad aproximada)"
[Esperá respuesta]
P4: "¿Cuáles fueron los puntos del orden del día?"
[Esperá respuesta]
P5: "¿Qué se acordó o resolvió en cada punto?"
[Esperá respuesta]
P6: "¿Hubo algún pedido, queja o situación especial a registrar? (si no, 'no')"
[Con las 6 respuestas → generá]

[TITULO] Acta N° (completar) — {tipo de reunión} — {fecha}
[SECCION] Datos de la reunión
[CAMPO] Institución: (completar)
[CAMPO] Tipo: {v}
[CAMPO] Fecha: {v}
[CAMPO] Hora de inicio: {v}
[CAMPO] Hora de cierre: (completar)
[CAMPO] Participantes: {v}
${campoDirectivo}
[SECCION] Orden del día
[LISTA] {punto tratado}
[SECCION] Desarrollo
[TEXTO] {tratamiento de cada punto — tono formal y objetivo, 3-5 oraciones por punto}
[SECCION] Acuerdos y resoluciones
[LISTA] {acuerdo o resolución concreta}
[SECCION] Cierre
[TEXTO] Sin más asuntos que tratar, se da por finalizada la reunión a las (completar) hs. Se labra la presente acta que, leída y hallada conforme, firman los presentes.
[CAMPO] Firma y sello directivo/a:
[CAMPO] Firma secretario/a:
[FIN_DOCUMENTO]
Después: "¡Lista el acta! 🗂️ ¿Ajustamos algo antes de descargar?"`;

    case 'comunicados': return base + `

FLUJO: COMUNICADO O CIRCULAR
Hacé UNA pregunta por turno. Esperá cada respuesta.
SIEMPRE empezá con P1 — aunque el usuario ya haya mencionado el tipo de comunicado.

P1: "¿Qué tipo de comunicación necesitás?
1. Comunicado general a todas las familias
2. Comunicado por grado o sección
3. Citación individual a una familia
4. Circular interna al personal"
[Esperá respuesta]
P2: "¿Cuál es el motivo principal?"
[Esperá respuesta]
P3: "¿Hay fechas, lugares u horarios concretos?"
[Esperá respuesta]
P4: "¿Se requiere alguna acción de las familias o el personal? (firmar, traer algo, confirmar — o 'no')"
[Con las 4 respuestas → generá según el tipo elegido]

COMUNICADO A FAMILIAS (tipos 1 y 2):
[TITULO] Comunicado — {escuela} — {fecha}
[SECCION] Estimadas familias:
[TEXTO] {motivo y contexto — 2-3 oraciones cálidas y directas}
[SECCION] Información
[LISTA] {dato clave: fecha, lugar, horario, requisito}
[SECCION] Se solicita
[TEXTO] {acción requerida — omitir sección si no aplica}
[SECCION] Cierre
[TEXTO] Ante consultas, comunicarse con secretaría. Agradecemos su colaboración.
${campoDirectivo}
[CAMPO] Fecha:
[FIN_DOCUMENTO]

CITACIÓN INDIVIDUAL (tipo 3):
[TITULO] Citación — {apellido familia} — {fecha}
[SECCION] Estimada familia:
[TEXTO] {motivo en tono profesional y no alarmante — 2 oraciones}
[CAMPO] Fecha y hora: {v}
[CAMPO] Lugar: Dirección de la escuela
[TEXTO] Rogamos confirmar asistencia o comunicar imposibilidad para reprogramar.
${campoDirectivo}
[FIN_DOCUMENTO]

CIRCULAR INTERNA (tipo 4):
[TITULO] Circular N° (completar) — {escuela} — {fecha}
[SECCION] Al personal:
[TEXTO] {motivo — 1-2 oraciones directas}
[SECCION] Se informa / Se solicita
[LISTA] {punto concreto}
[CAMPO] Fecha límite: {v si aplica}
${campoDirectivo}
[FIN_DOCUMENTO]

Después: "¡Listo! 🗂️ ¿Ajustamos algo antes de descargar?"`;

    case 'proyectos': return base + `

FLUJO: PROYECTO INSTITUCIONAL
Proyectos específicos, acotados y viables — enmarcados en el PEI cordobés.
Hacé UNA pregunta por turno. Esperá cada respuesta.
SIEMPRE empezá con P1 — aunque el usuario ya haya mencionado el tipo de proyecto.

P1: "¿Qué tipo de proyecto necesitás?
1. Mejora de aprendizajes (lectura, matemática, etc.)
2. Feria institucional (Ciencias, Libro, Arte)
3. Convivencia o ESI
4. Articulación con la comunidad
5. Otro — describí"
[Esperá respuesta]
P2: "¿Cuál es el problema o necesidad que origina el proyecto?"
[Esperá respuesta]
P3: "¿A quiénes va dirigido y cuánto dura?"
[Esperá respuesta]
P4: "¿Qué recursos humanos y materiales tenés disponibles?"
[Esperá respuesta]
P5: "¿Hay alguna fecha o evento institucional que anclé el proyecto?"
[Con las 5 respuestas → generá]

[TITULO] Proyecto: {nombre} — {escuela} — {año}
[SECCION] Identificación
[CAMPO] Institución: (completar)
${campoDirectivo}
[CAMPO] Tipo: {v}
[CAMPO] Destinatarios: {v}
[CAMPO] Duración: {v}
[CAMPO] Fecha de inicio: {v}
[SECCION] Fundamentación
[TEXTO] {diagnóstico institucional que origina el proyecto — 3-4 oraciones}
[SECCION] Propósitos
[LISTA] {propósito concreto y observable — 4-5 ítems}
[SECCION] Objetivos específicos
[LISTA] {objetivo medible — 4-5 ítems}
[SECCION] Contenidos y ejes
[LISTA] {eje de trabajo articulado con DC Córdoba si aplica}
[SECCION] Acciones y cronograma
[LISTA] {acción concreta — responsable — mes/trimestre}
[SECCION] Recursos
[SUBSECCION] Humanos
[LISTA] {docentes, directivos, familias, comunidad}
[SUBSECCION] Materiales y digitales
[LISTA] {recurso concreto}
[SECCION] Evaluación
[TEXTO] {indicadores de proceso y resultado — cómo se evaluará el impacto}
[SECCION] Articulación con el PEI
[TEXTO] {cómo se vincula con el proyecto educativo institucional}
[FIN_DOCUMENTO]
Después: "¡Proyecto armado! 🗂️ ¿Ajustamos algo antes de descargar?"`;

    default: return base + `

FLUJO: CONSULTA DE GESTIÓN
Sos la directora más experimentada de la región.
Ayudás con: normativa educativa cordobesa, conflictos con docentes o familias, licencias, sanciones, acuerdos de convivencia, convivencia escolar, prevención de adicciones, liderazgo pedagógico, autoevaluación institucional, relación con supervisores, comunicación institucional.
Si la consulta es sobre licencias específicas, remitís al chip "Licencias docentes" para orientación normativa precisa.
Respondés con pasos concretos. Citás normativa SOLO si está en el contexto RAG o si es una norma que conocés con certeza absoluta (Ley 9870, Ley 26.150). En caso de duda, decís "verificá el artículo exacto con supervisión".
Si la situación es delicada: primero validás, después orientás.

PROPUESTAS CREATIVAS — aplicar cuando la situación lo amerita:
Cuando aplique, agregá al final:
"💡 *Una idea:* {propuesta concreta, práctica y original — 2-3 oraciones. Implementable esta semana sin recursos extras.}"

Cerrás con una pregunta o invitación breve.
Si el directivo quiere generar un documento: indicale los botones del menú inferior.`;
  }
}

// ── Infraestructura ─────────────────────────────────────────────

const CONFIG = {
  RATE_LIMIT_REQUESTS: 20,
  RATE_LIMIT_WINDOW:   60,
  MAX_MESSAGE_LENGTH:  2000,
  REQUEST_TIMEOUT:     30000,
};

const ALLOWED_ORIGINS = [
  'https://abfacundotorres.github.io',
  'https://norma.aulaclara.com.ar',
  'https://www.norma.aulaclara.com.ar',
];

function getAllowedOrigin(origin) {
  if (!origin) return null;
  return ALLOWED_ORIGINS.find(o => origin.startsWith(o)) ? origin : null;
}

function corsHeaders(origin) {
  return {
    'Access-Control-Allow-Origin':  getAllowedOrigin(origin) || 'null',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'X-Content-Type-Options':       'nosniff',
    'X-Frame-Options':              'DENY',
  };
}

function respond(body, status, origin) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...corsHeaders(origin) },
  });
}

async function checkRateLimit(ip, kv) {
  if (!kv) return true;
  const key = `rl:${ip}`;
  const now = Math.floor(Date.now()/1000);
  let d = {count:0,window:now};
  try { const r=await kv.get(key); if(r) d=JSON.parse(r); } catch { return true; }
  if (now-d.window >= CONFIG.RATE_LIMIT_WINDOW) d={count:1,window:now};
  else d.count++;
  try { await kv.put(key,JSON.stringify(d),{expirationTtl:CONFIG.RATE_LIMIT_WINDOW}); } catch {}
  return d.count <= CONFIG.RATE_LIMIT_REQUESTS;
}

function sanitizeHistory(raw) {
  if (!Array.isArray(raw)) return [];
  const filtered = raw
    .slice(-HISTORY_LIMIT)
    .filter(t => t && typeof t==='object' && (t.role==='user'||t.role==='model') && typeof t.parts?.[0]?.text==='string')
    .map(t => {
      const text = String(t.parts[0].text);
      if (isDocumentText(text)) {
        const tm = text.match(/\[TITULO\]\s*([^\n]+)/);
        const collapsed = tm ? `[documento generado: ${tm[1].trim()}]` : '[documento generado]';
        return { role: t.role, parts: [{ text: collapsed }] };
      }
      const clean = t.role === 'model'
        ? text.trim().slice(0, 600)
        : text.slice(0, 800);
      return { role: t.role, parts: [{ text: clean }] };
    });
  // Garantizar alternancia user/model
  const alt = []; let last = null;
  for (const t of filtered) { if (t.role !== last) { alt.push(t); last = t.role; } }
  const first = alt.findIndex(t => t.role === 'user');
  return first > 0 ? alt.slice(first) : alt;
}

function toGroqMessages(contents, systemPrompt) {
  const msgs = [{ role: 'system', content: systemPrompt }];
  for (const t of contents) {
    const text = t.parts?.[0]?.text || '';
    if (text) msgs.push({ role: t.role === 'model' ? 'assistant' : 'user', content: text });
  }
  return msgs;
}

function extractGroq(data) {
  const c = data.choices?.[0], t = c?.message?.content;
  if (!t) return { error: data.error?.message || 'Respuesta vacía.' };
  if (c?.finish_reason === 'length') return { reply: t + '\n\n_(Escribime "continuá" 😊)_', model: data.model };
  return { reply: t, model: data.model };
}

function extractGemini(data) {
  if (data.promptFeedback?.blockReason) return { error: 'Consulta bloqueada.' };
  const c = data.candidates?.[0], t = c?.content?.parts?.[0]?.text;
  if (!t) return { error: 'Respuesta vacía del servicio de respaldo.' };
  if (c?.finishReason === 'MAX_TOKENS') return { reply: t + '\n\n_(Escribime "continuá" 😊)_', model: 'gemini-flash-lite' };
  return { reply: t, model: 'gemini-flash-lite' };
}

async function fetchWithAbort(url, opts, timeout) {
  const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), timeout);
  try { const res = await fetch(url, { ...opts, signal: ctrl.signal }); clearTimeout(timer); return res; }
  catch(err) { clearTimeout(timer); if (err.name === 'AbortError') throw new Error('timeout'); throw new Error('fetch_failed'); }
}

async function callLLM(contents, prompt, env, maxTokens, temperature) {
  if (env.GROQ_API_KEY) {
    try {
      const res = await fetchWithAbort(GROQ_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${env.GROQ_API_KEY}` },
        body: JSON.stringify({ model: GROQ_MODEL, messages: toGroqMessages(contents, prompt), temperature, max_tokens: maxTokens, top_p: 0.9 }),
      }, CONFIG.REQUEST_TIMEOUT);
      if (res.ok) { const r = extractGroq(await res.json()); if (!r.error) return r; }
      else if (res.status !== 429 && res.status >= 400 && res.status < 500)
        return { error: 'Error de configuración. Contactá al administrador.' };
    } catch(err) { console.warn('[Groq]', err.message); }
  }
  if (!env.GEMINI_API_KEY) return { error: 'Estoy con mucha demanda ahora. Intentá en unos minutos.' };
  try {
    const res = await fetchWithAbort(`${GEMINI_URL}?key=${env.GEMINI_API_KEY}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: prompt }] },
        contents: contents.map(t => ({ role: t.role, parts: [{ text: t.parts[0].text }] })),
        generationConfig: { temperature, maxOutputTokens: maxTokens, topP: 0.9 },
        safetySettings: [
          { category: 'HARM_CATEGORY_HARASSMENT',        threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
          { category: 'HARM_CATEGORY_HATE_SPEECH',       threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
          { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
          { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
        ],
      }),
    }, CONFIG.REQUEST_TIMEOUT);
    if (!res.ok) return { error: 'Estoy con mucha demanda ahora. Intentá en unos minutos.' };
    return extractGemini(await res.json());
  } catch(err) {
    return { error: err.message === 'timeout' ? 'Tardé demasiado. Intentá de nuevo.' : 'Sin conexión. Revisá tu red.' };
  }
}

// ── Handler ──────────────────────────────────────────────────────

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get('Origin') || '';
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders(origin) });
    if (request.method !== 'POST') return respond({ error: 'Método no permitido.' }, 405, origin);

    const ip = request.headers.get('CF-Connecting-IP');
    if (!ip) return respond({ error: 'Request inválido.' }, 400, origin);

    if (!await checkRateLimit(ip, env.RATE_LIMIT_KV))
      return respond({ error: 'Demasiadas consultas seguidas. Esperá un minuto.' }, 429, origin);

    let body;
    try { body = await request.json(); }
    catch { return respond({ error: 'Request inválido.' }, 400, origin); }

    const { message, history, pin, flow: clientFlow, ctx: rawCtx, chip_usado, problema_detectado, session_id, _survey, tipo } = body;

    // ── Encuesta de satisfacción o evento de descarga ──
    if (_survey === true || (_survey === false && tipo === 'evento' && body.doc_descargado)) {
      if (env.METRICS_WEBHOOK) {
        try {
          await fetch(env.METRICS_WEBHOOK, {
            method: 'POST',
            redirect: 'follow',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...body, _survey: undefined }),
          });
        } catch(e) { console.error('[survey proxy]', e.message); }
      }
      return respond({ ok: true }, 200, origin);
    }

    const docenteCtx = (rawCtx && typeof rawCtx === 'object' && !Array.isArray(rawCtx)) ? rawCtx : {};

    if (env.MUNICIPIO_PINS?.trim()) {
      const pines = env.MUNICIPIO_PINS.split(',').map(p => p.trim().toUpperCase()).filter(Boolean);
      if (pines.length && (!pin || !pines.includes(String(pin).trim().toUpperCase())))
        return respond({ error: 'Código de acceso inválido. Pedíselo a tu coordinador.' }, 403, origin);
    }

    if (!message || typeof message !== 'string' || !message.trim())
      return respond({ error: 'Mensaje inválido.' }, 400, origin);
    const clean = message.trim();
    if (clean.length > CONFIG.MAX_MESSAGE_LENGTH) return respond({ error: 'Mensaje demasiado largo.' }, 400, origin);

    const flow = VALID_FLOWS.has(clientFlow) ? clientFlow : 'general';

    const { context: ragContext, count: ragChunks, error: ragError } = flow === 'resumen'
      ? { context: '', count: 0, error: false }
      : await retrieveContext(clean, env, flow);

    const contents = [...sanitizeHistory(history), { role: 'user', parts: [{ text: clean }] }];
    const prompt   = buildPrompt(flow, docenteCtx, ragContext);
    const { maxTokens, temperature } = getLLMParams(flow);

    const result = await callLLM(contents, prompt, env, maxTokens, temperature);
    if (result.error) return respond({ error: result.error }, 502, origin);

    logEvent(ctx, env, {
      flow,
      municipio:          pin ? String(pin).toUpperCase().slice(0, 20) : 'demo',
      rag_usado:          ragChunks > 0,
      rag_chunks:         ragChunks,
      rag_error:          ragError,
      model_used:         result.model || 'unknown',
      chip_usado:         chip_usado    || '',
      problema_detectado: problema_detectado || '',
      session_id:         session_id    || '',
    });

    return respond({ reply: result.reply, flow }, 200, origin);
  },
};
