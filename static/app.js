// ===== elementos =====
const dropzone = document.getElementById('dropzone');
const dropContent = document.getElementById('dropContent'); // dentro do dropzone
const fileInput = document.getElementById('file');
const textArea = document.getElementById('text');
const analyzeBtn = document.getElementById('analyze');

const result = document.getElementById('result');
const badge = document.getElementById('badge');
const chips = document.getElementById('signals');
const overridesBox = document.getElementById('overrides');
const reply = document.getElementById('reply');
const meta = document.getElementById('meta');

const confVal = document.getElementById('confVal');
const confBar = document.getElementById('confBar');
const rawJson = document.getElementById('rawJson');
const toggleJsonBtn = document.getElementById('toggleJson');

const loader = document.getElementById('loader');
const toast = document.getElementById('toast');

const copyBtn = document.getElementById('copy');
const copyClearBtn = document.getElementById('copyClear');

let droppedFile = null;

// ===== helpers =====
function showToast(msg) {
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 1800);
}

function mkChip(text, extraClass = '') {
  const el = document.createElement('span');
  el.className = `chip ${extraClass}`.trim();
  el.textContent = text;
  return el;
}

// ===== dropzone: modos =====
function bindPickButton() {
  // reatribui o listener do "clique para selecionar" sempre que recriamos o conteúdo
  document.getElementById('pick')?.addEventListener('click', () => fileInput.click());
}

function setDropzoneFile(name) {
  dropzone.classList.add('loaded');
  dropContent.innerHTML = `
    <div>
      <strong>📄 ${name}</strong>
      <div class="file-label">Arquivo selecionado</div>
      <button type="button" id="clearFile" class="link">Remover</button>
    </div>
  `;
  document.getElementById('clearFile')?.addEventListener('click', () => {
    droppedFile = null;
    fileInput.value = '';
    resetDropzone();
  });
}

function resetDropzone() {
  dropzone.classList.remove('loaded');
  dropContent.innerHTML = `
    <div>
      <strong>Arraste</strong> um arquivo .txt/.pdf aqui ou
      <button id="pick" type="button" class="link">clique para selecionar</button>
    </div>
  `;
  bindPickButton();
}

// ===== eventos: input file & drag/drop =====
bindPickButton();

fileInput?.addEventListener('change', (e) => {
  droppedFile = e.target.files[0] || null;
  if (droppedFile) {
    showToast(`Arquivo selecionado: ${droppedFile.name}`);
    setDropzoneFile(droppedFile.name);
  } else {
    resetDropzone();
  }
});

['dragenter','dragover'].forEach(ev => dropzone?.addEventListener(ev, (e)=>{
  e.preventDefault(); e.stopPropagation(); dropzone.classList.add('drag');
}));
['dragleave','drop'].forEach(ev => dropzone?.addEventListener(ev, (e)=>{
  e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('drag');
}));

dropzone?.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f) return;
  if (!/\.(txt|pdf)$/i.test(f.name)) { 
    showToast('Envie .txt ou .pdf'); 
    return; 
  }
  droppedFile = f;
  showToast(`Arquivo solto: ${f.name}`);
  setDropzoneFile(f.name);
});

// ===== chamada API =====
async function analyze() {
  analyzeBtn.disabled = true;
  analyzeBtn.setAttribute('aria-busy', 'true');
  if (loader) loader.classList.remove('hidden');
  const originalLabel = analyzeBtn.textContent;
  analyzeBtn.textContent = 'Processando…';

  try {
    const fd = new FormData();
    if (droppedFile) {
      fd.append('email_file', droppedFile);
    } else {
      const txt = (textArea.value || '').trim();
      if (!txt) { showToast('Cole um texto ou envie um arquivo.'); return; }
      fd.append('email_text', txt);
    }

    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error('Resposta inválida do servidor');
    }
    if (!res.ok) {
      const msg = (data && (data.detail || data.message)) || 'Falha ao processar';
      throw new Error(msg);
    }

    renderResult(data);
  } catch (e) {
    showToast(e.message || 'Erro ao processar');
    console.error(e);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.setAttribute('aria-busy', 'false');
    if (loader) loader.classList.add('hidden');
    analyzeBtn.textContent = originalLabel || 'Analisar';
  }
}

function renderResult(data) {
  result.classList.remove('hidden');

  // categoria + badge
  badge.textContent = data.category || '—';
  badge.classList.remove('success','neutral');
  badge.classList.add(data.category === 'Produtivo' ? 'success' : 'neutral');

  // confiança (valor + barra)
  const pct = Math.round((data.confidence || 0) * 100);
  if (confVal) confVal.textContent = isFinite(pct) ? `${pct}%` : '—';
  if (confBar) confBar.style.width = isFinite(pct) ? `${pct}%` : '0%';

  // sinais
  chips.innerHTML = '';
  (data.meta?.signals || []).forEach(s => chips.appendChild(mkChip(s)));

  // overrides (apenas os true)
  if (overridesBox) {
    overridesBox.innerHTML = '';
    const ov = data.meta?.overrides || {};
    const labels = {
      gratitude_no_action: 'Agradecimento',
      action_over_low_conf: 'Ação baixa conf.',
      marketing_newsletter: 'Marketing/Newsletter',
      resolved_or_cancelled: 'Resolvido/Cancelado',
      urgency_boost: 'Urgência',
      short_question_hint: 'Pergunta curta'
    };
    Object.entries(ov).forEach(([k,v]) => {
      if (v === true) overridesBox.appendChild(mkChip(labels[k] || k, 'override'));
    });
  }

  // resposta
  reply.value = data.reply || '';

  // json bruto
  if (rawJson) rawJson.textContent = JSON.stringify(data, null, 2);

  // meta
  meta.textContent =
    `Usou HF: ${data.meta?.used_hf ? 'sim' : 'não'} | Usou LLM: ${data.meta?.used_openai ? 'sim' : 'não'} | Fallbacks: ${(data.meta?.fallbacks||[]).join(', ')}`;
}

// ===== ações =====
analyzeBtn?.addEventListener('click', analyze);

copyBtn?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(reply.value || '');
    showToast('Resposta copiada!');
  } catch {
    showToast('Não foi possível copiar');
  }
});

copyClearBtn?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(reply.value || '');
  } catch { /* ignora se não conseguir copiar */ }
  textArea.value = '';
  droppedFile = null;
  if (fileInput) fileInput.value = '';
  resetDropzone();
  showToast('Copiado e limpo!');
});

// toggle JSON bruto
toggleJsonBtn?.addEventListener('click', () => {
  if (!rawJson) return;
  rawJson.classList.toggle('hidden');
});

// Atalho: Ctrl+Enter para enviar
textArea.addEventListener("keydown", (e) => {
    const isMac = navigator.platform.toUpperCase().includes('MAC');
    const cmdEnter = isMac && e.metaKey && e.key === "Enter";
    const ctrlEnter = !isMac && e.ctrlKey && e.key === "Enter";
    if (cmdEnter || ctrlEnter) {
        e.preventDefault();
        analyze();
    }
});

// Ajusta tooltip do botão Analisar conforme o sistema
(function () {
    const analyzeBtn = document.getElementById('analyze');
    if (!analyzeBtn) return;
    const isMac = navigator.platform.toUpperCase().includes('MAC');
    analyzeBtn.title = isMac ? "Atalho: ⌘ + Enter" : "Atalho: Ctrl + Enter";
})();


