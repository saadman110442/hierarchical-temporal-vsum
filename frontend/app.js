// app.js — frontend logic for the video summarization demo
// Vanilla JS, no build step. Uses fetch + polling.
// Supports both Part 1 (HiT-UNet, visual) and Part 2 (MM-HiT-UNet, multimodal).

(() => {
    'use strict';

    // ---------- DOM ----------
    const $ = (id) => document.getElementById(id);

    const dropzone   = $('dropzone');
    const fileInput  = $('file-input');
    const uploadSec  = $('upload-section');
    const progSec    = $('progress-section');
    const errorSec   = $('error-section');
    const resultsSec = $('results-section');

    const progMessage = $('progress-message');
    const progPercent = $('progress-percent');
    const progBar     = $('progress-bar');
    const progFill    = $('progress-fill');
    const progMeta    = $('progress-meta');
    const stagePills  = $('stage-pills');

    const captionsWrap = $('captions-wrap');
    const captionsFeed = $('captions-feed');

    const errorMsg  = $('error-message');
    const resetErr  = $('reset-after-error');

    const vOrig       = $('video-original');
    const vSumm       = $('video-summary');
    const downloadBtn = $('download-btn');
    const resetBtn    = $('reset-btn');
    const summaryTitle       = $('summary-title');
    const resultVariantBadge = $('result-variant-badge');
    const sampleCaptionsWrap = $('sample-captions-wrap');
    const sampleCaptions     = $('sample-captions');

    // Part 2 visualisation containers
    const vizPart2Wrap    = $('viz-part2-wrap');
    const vizPart2Body    = $('viz-part2-body');
    const vizToggleBtn    = $('viz-toggle');
    const vizToggleLabel  = $('viz-toggle-label');
    const wordcloudEl     = $('wordcloud');
    const scatterEl       = $('scatter');
    const scatterVariance = $('scatter-variance');
    const shotRibbonEl    = $('shot-ribbon');
    const shotRibbonList  = $('shot-ribbon-list');

    // Stashed Part 2 viz data for the current result. The viz isn't rendered
    // until the user clicks "Show caption visualisations".
    let vizCachedData = null;
    let vizRendered = false;

    const statDur    = $('stat-duration');
    const statSumm   = $('stat-summary-duration');
    const statRatio  = $('stat-ratio');
    const statShots  = $('stat-shots');

    const variantCards = document.querySelectorAll('.variant-card');
    const demoVariantLabel = $('demo-variant-label');
    const p2Unavailable = $('p2-unavailable');

    const healthP1Dot  = $('health-p1-dot');
    const healthP1Text = $('health-p1-text');
    const healthP2Dot  = $('health-p2-dot');
    const healthP2Text = $('health-p2-text');
    const healthDevice = $('health-device');

    const copyBibtex   = $('copy-bibtex');
    const copyFeedback = $('copy-feedback');
    const bibtexTabs   = document.querySelectorAll('.bibtex-tab');
    const bibtexBodies = document.querySelectorAll('.bibtex-body');

    // ---------- State ----------
    let selectedVariant = 'part1';
    let serverPart1Loaded = false;
    let serverPart2Loaded = false;
    let selectedDevice = 'auto';     // user's UI choice: auto | cpu | cuda
    let cudaAvailable = false;

    // ---------- Pipeline stages ----------
    // load_multimodal is Part 2 only (lazy CLIP+BLIP-2 load)
    const STAGES_PART1 = [
        { key: 'upload',           label: 'Upload' },
        { key: 'extract_features', label: 'Features' },
        { key: 'kts',              label: 'Shots (KTS)' },
        { key: 'model',            label: 'Model' },
        { key: 'knapsack',         label: 'Keyshots' },
        { key: 'encode',           label: 'Encode' },
        { key: 'reencode',         label: 'Finalize' },
    ];
    const STAGES_PART2 = [
        { key: 'upload',           label: 'Upload' },
        { key: 'load_multimodal',  label: 'Load CLIP+BLIP-2' },
        { key: 'extract_features', label: 'Features' },
        { key: 'kts',              label: 'Shots (KTS)' },
        { key: 'model',            label: 'Model' },
        { key: 'knapsack',         label: 'Keyshots' },
        { key: 'encode',           label: 'Encode' },
        { key: 'reencode',         label: 'Finalize' },
    ];

    function currentStages() {
        return selectedVariant === 'part2' ? STAGES_PART2 : STAGES_PART1;
    }

    function renderPills(currentStage, status) {
        stagePills.innerHTML = '';
        const stages = currentStages();
        const currentIdx = stages.findIndex(s => s.key === currentStage);
        stages.forEach((s, i) => {
            const pill = document.createElement('div');
            pill.className = 'stage-pill';
            if (status === 'done') {
                pill.classList.add('done');
            } else if (i < currentIdx) {
                pill.classList.add('done');
            } else if (i === currentIdx) {
                pill.classList.add('active');
            }
            pill.innerHTML = `<span class="dot"></span>${s.label}`;
            stagePills.appendChild(pill);
        });
    }

    // ---------- Helpers ----------
    function fmtTime(seconds) {
        if (!isFinite(seconds) || seconds < 0) return '—';
        const s = Math.round(seconds);
        const m = Math.floor(s / 60);
        const r = s % 60;
        return `${m}:${String(r).padStart(2, '0')}`;
    }
    function fmtPct(p) { return `${Math.round(p * 100)}%`; }
    function show(el)   { el && el.classList.remove('hidden'); }
    function hide(el)   { el && el.classList.add('hidden'); }

    function variantColorClasses() {
        // Applied to dropzone, progress bar, stage-pills container — lets CSS
        // swap the accent colour (terracotta ↔ indigo) for Part 1 vs Part 2.
        const p2 = selectedVariant === 'part2';
        dropzone.classList.toggle('variant-part2', p2);
        progBar.classList.toggle('variant-part2', p2);
        stagePills.parentElement.classList.toggle('variant-part2', p2);
    }

    function variantBadgeHTML(variant, variant_label) {
        const p2 = variant === 'part2';
        const label = variant_label || (p2 ? 'MM-HiT-UNet (Part 2)' : 'HiT-UNet (Part 1)');
        const cls = p2 ? 'chip chip-p2' : 'chip chip-p1';
        return `<span class="${cls}"><span class="dot"></span>${label}</span>`;
    }

    function updateDemoVariantLabel() {
        const p2 = selectedVariant === 'part2';
        demoVariantLabel.innerHTML = p2
            ? `<span class="chip chip-p2"><span class="dot"></span>Running: MM-HiT-UNet (Part 2, CLIP + BLIP-2 multimodal)</span>`
            : `<span class="chip chip-p1"><span class="dot"></span>Running: HiT-UNet (Part 1, visual-only)</span>`;
    }

    // ---------- Variant switching ----------
    // The switcher now always swaps the *displayed* variant (abstract, method,
    // pipeline diagram, BibTeX, demo badge) so you can read either paper even
    // before the server has best_part2.pt. Submission-time gating lives inside
    // handleFile(), which refuses Part 2 uploads if the server hasn't loaded
    // the Part 2 model.
    function setVariant(v) {
        if (v !== 'part1' && v !== 'part2') v = 'part1';
        selectedVariant = v;

        variantCards.forEach(card => {
            const is = card.dataset.variant === v;
            card.classList.toggle('active', is);
            card.setAttribute('aria-pressed', is ? 'true' : 'false');
        });

        // Swap abstract + method + citation by toggling `hidden` directly on
        // every element that carries a variant id (more defensive than the
        // previous `.variant-content` broad sweep — makes the switch robust
        // even if new content sections get added later).
        ['abstract', 'method', 'bibtex'].forEach((section) => {
            const keep = $(`${section}-${v}`);
            const drop = $(`${section}-${v === 'part1' ? 'part2' : 'part1'}`);
            if (drop) drop.classList.add('hidden');
            if (keep) {
                keep.classList.remove('hidden');
                keep.classList.remove('swap-in');
                // Trigger animation restart
                void keep.offsetWidth;
                keep.classList.add('swap-in');
            }
        });

        // Swap citation tab buttons
        bibtexTabs.forEach(t => t.classList.toggle('active', t.dataset.tab === v));

        updateDemoVariantLabel();
        variantColorClasses();
    }

    // Event delegation on the container, so clicks on any descendant of a
    // .variant-card (ribbon, watermark, paragraph text, the chips, etc.) still
    // resolve to the card itself. Per-element listeners were occasionally
    // missing clicks that landed on the absolutely-positioned ribbon.
    const variantContainer = document.getElementById('variant-cards');

    function handleVariantPick(card) {
        if (!card) return;
        const v = card.dataset.variant;
        if (v !== 'part1' && v !== 'part2') return;
        setVariant(v);

        // If the user picked Part 2 but the server doesn't have it, pulse the
        // inline warning so they see why upload will refuse.
        if (v === 'part2' && !serverPart2Loaded && p2Unavailable) {
            p2Unavailable.classList.remove('hidden');
            try {
                p2Unavailable.animate(
                    [{ opacity: 0.3 }, { opacity: 1 }],
                    { duration: 400, iterations: 1 }
                );
            } catch {}
        }
    }

    if (variantContainer) {
        variantContainer.addEventListener('click', (e) => {
            const card = e.target.closest('.variant-card');
            if (!card || !variantContainer.contains(card)) return;
            e.preventDefault();
            handleVariantPick(card);
        });
        // Keyboard activation (Enter / Space) when the card has focus
        variantContainer.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const card = e.target.closest('.variant-card');
            if (!card) return;
            e.preventDefault();
            handleVariantPick(card);
        });
    } else {
        // Fallback: per-card listener if the container id changes in future
        variantCards.forEach(card => {
            card.addEventListener('click', (e) => {
                e.preventDefault();
                handleVariantPick(card);
            });
        });
    }

    // ---------- Device picker (CPU / GPU) ----------
    const deviceOpts   = document.querySelectorAll('.device-opt');
    const gpuBtn       = document.getElementById('device-gpu-btn');
    const gpuNameEl    = document.getElementById('device-gpu-name');

    function setDevice(dev) {
        if (!['auto', 'cpu', 'cuda'].includes(dev)) dev = 'auto';
        // Don't let the user pick GPU if it's not available
        if (dev === 'cuda' && !cudaAvailable) {
            dev = 'cpu';
        }
        selectedDevice = dev;
        deviceOpts.forEach(b => {
            const on = b.dataset.device === dev;
            b.classList.toggle('active', on);
            b.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    deviceOpts.forEach(b => {
        b.addEventListener('click', (e) => {
            if (b.disabled) return;
            e.preventDefault();
            setDevice(b.dataset.device);
        });
    });

    // ---------- Part 2 viz toggle ----------
    if (vizToggleBtn) {
        vizToggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (!vizCachedData || !vizPart2Body) return;
            const wasHidden = vizPart2Body.classList.contains('hidden');
            if (wasHidden) {
                // Lazy-render on first reveal so we don't pay the SVG/dom cost
                // unless the user actually wants to see it.
                if (!vizRendered) {
                    try {
                        renderWordCloud(vizCachedData.captions);
                        renderScatter(
                            vizCachedData.points2d,
                            vizCachedData.selected,
                            vizCachedData.captions,
                            vizCachedData.variance,
                        );
                        renderShotRibbon(
                            vizCachedData.captions,
                            vizCachedData.shotIds,
                            vizCachedData.selected,
                        );
                        vizRendered = true;
                    } catch (err) {
                        console.warn('[viz] Part 2 visualisation failed:', err);
                    }
                }
                vizPart2Body.classList.remove('hidden');
                if (vizToggleLabel) vizToggleLabel.textContent = 'Hide caption visualisations';
                // Scroll the panel into view so the user actually sees it appear
                vizPart2Body.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                vizPart2Body.classList.add('hidden');
                if (vizToggleLabel) vizToggleLabel.textContent = 'Show caption visualisations';
            }
        });
    }

    // Bibtex tab clicks
    bibtexTabs.forEach(t => {
        t.addEventListener('click', () => {
            const v = t.dataset.tab;
            bibtexTabs.forEach(x => x.classList.toggle('active', x === t));
            bibtexBodies.forEach(b => b.classList.toggle('hidden', b.id !== `bibtex-${v}`));
        });
    });

    // ---------- Health check ----------
    async function checkHealth() {
        try {
            const r = await fetch('/api/health');
            const j = await r.json();

            serverPart1Loaded = !!j.part1_loaded;
            serverPart2Loaded = !!j.part2_loaded;

            // Part 1 dot
            healthP1Dot.className = 'inline-block w-2 h-2 rounded-full ' +
                (serverPart1Loaded ? 'bg-orange-600' : 'bg-amber-500');
            healthP1Text.textContent = serverPart1Loaded ? 'part 1: ready' : 'part 1: unloaded';

            // Part 2 dot
            healthP2Dot.className = 'inline-block w-2 h-2 rounded-full ' +
                (serverPart2Loaded ? 'bg-indigo-600' : 'bg-ink-mute');
            healthP2Text.textContent = serverPart2Loaded
                ? (j.mm_extractor_warm ? 'part 2: warm' : 'part 2: ready')
                : 'part 2: unloaded';

            // Device
            const device = (j.device || '').toLowerCase();
            const devLabel = device.startsWith('cuda') ? 'GPU' : (device === 'unloaded' ? '—' : 'CPU');
            healthDevice.textContent = `· ${devLabel}`;

            // CUDA availability — drives the GPU button in the device picker
            const cuda = j.cuda || {};
            cudaAvailable = !!cuda.available;
            if (gpuBtn) {
                gpuBtn.disabled = !cudaAvailable;
                if (cudaAvailable && Array.isArray(cuda.devices) && cuda.devices.length > 0) {
                    const d0 = cuda.devices[0];
                    const shortName = (d0.name || 'GPU')
                        .replace(/^NVIDIA\s+GeForce\s+/i, '')
                        .replace(/^NVIDIA\s+/i, '');
                    const free = d0.vram_free_mb;
                    const total = d0.vram_total_mb;
                    gpuBtn.title = `${d0.name} · ${free} / ${total} MB free`;
                    if (gpuNameEl) {
                        gpuNameEl.textContent = `· ${shortName}`;
                    }
                } else {
                    gpuBtn.title = 'No CUDA device detected. Install a CUDA build of torch (see setup_cuda.ps1).';
                    if (gpuNameEl) gpuNameEl.textContent = '· n/a';
                }
            }
            // If the user had picked GPU but it disappeared, fall back to auto
            if (selectedDevice === 'cuda' && !cudaAvailable) setDevice('auto');

            // Part 2 card availability
            const p2Card = document.querySelector('.variant-card[data-variant="part2"]');
            if (p2Card) {
                p2Card.setAttribute('aria-disabled', serverPart2Loaded ? 'false' : 'true');
            }
            if (p2Unavailable) {
                p2Unavailable.classList.toggle('hidden', serverPart2Loaded);
            }

            // Preview is fine even without Part 2 loaded — only submission is gated.
            // (We intentionally do NOT bounce the UI back to Part 1 here.)
        } catch (e) {
            healthP1Dot.className = 'inline-block w-2 h-2 rounded-full bg-red-500';
            healthP1Text.textContent = 'part 1: offline';
            healthP2Dot.className = 'inline-block w-2 h-2 rounded-full bg-red-500';
            healthP2Text.textContent = 'part 2: offline';
            healthDevice.textContent = '';
        }
    }
    checkHealth();
    // refresh health every 10s so it reflects model warm-up
    setInterval(checkHealth, 10000);

    // ---------- Drag and drop ----------
    ['dragenter', 'dragover'].forEach(ev =>
        dropzone.addEventListener(ev, (e) => {
            e.preventDefault(); e.stopPropagation();
            dropzone.classList.add('hover');
        })
    );
    ['dragleave', 'drop'].forEach(ev =>
        dropzone.addEventListener(ev, (e) => {
            e.preventDefault(); e.stopPropagation();
            dropzone.classList.remove('hover');
        })
    );
    dropzone.addEventListener('drop', (e) => {
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) handleFile(f);
    });
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) handleFile(f);
    });

    // ---------- Upload + polling ----------
    async function handleFile(file) {
        if (!file.type.startsWith('video/') && !/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name)) {
            showError(`Not a video file: ${file.name}`);
            return;
        }

        // Defensive: re-check Part 2 availability right before upload
        if (selectedVariant === 'part2' && !serverPart2Loaded) {
            showError('Part 2 model is not loaded on the server. Please add best_part2.pt to backend/models/ and restart.');
            return;
        }

        hide(uploadSec); hide(errorSec); hide(resultsSec);
        show(progSec);
        progMessage.textContent = 'Uploading…';
        progPercent.textContent = '0%';
        progFill.style.width = '0%';
        progMeta.textContent = `${file.name} · ${(file.size / (1024*1024)).toFixed(1)} MB · ${selectedVariant === 'part2' ? 'MM-HiT-UNet' : 'HiT-UNet'}`;

        // Reset/hide captions from previous run
        captionsFeed.innerHTML = '';
        captionsWrap.classList.toggle('hidden', selectedVariant !== 'part2');

        renderPills('upload', 'running');
        variantColorClasses();

        const form = new FormData();
        form.append('file', file);
        form.append('variant', selectedVariant);
        form.append('device', selectedDevice);

        let jobId;
        try {
            jobId = await uploadWithProgress(form, (p) => {
                progFill.style.width = `${Math.round(p * 30)}%`; // upload ~ 0..30% of overall
                progPercent.textContent = fmtPct(p * 0.30);
            });
        } catch (e) {
            showError(e.message || String(e));
            return;
        }

        try {
            const final = await pollStatus(jobId);
            renderResults(jobId, final);
        } catch (e) {
            showError(e.message || String(e));
        }
    }

    function uploadWithProgress(form, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/summarize');
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) onProgress(e.loaded / e.total);
            };
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const j = JSON.parse(xhr.responseText);
                        resolve(j.job_id);
                    } catch (err) {
                        reject(new Error('Malformed server response'));
                    }
                } else {
                    let msg = `Upload failed (HTTP ${xhr.status})`;
                    try {
                        const j = JSON.parse(xhr.responseText);
                        if (j.detail) msg = j.detail;
                    } catch {}
                    reject(new Error(msg));
                }
            };
            xhr.onerror = () => reject(new Error('Network error during upload'));
            xhr.send(form);
        });
    }

    // Track the last caption shown so we don't duplicate lines that repeat
    let lastCaption = null;

    function maybePushCaption(txt) {
        if (!txt) return;
        if (txt === lastCaption) return;
        lastCaption = txt;
        const span = document.createElement('span');
        span.className = 'cap';
        span.textContent = txt;
        captionsFeed.appendChild(span);
        // Keep only the last ~30 captions to avoid DOM growth
        while (captionsFeed.children.length > 30) captionsFeed.removeChild(captionsFeed.firstChild);
        // Auto-scroll
        captionsFeed.scrollTop = captionsFeed.scrollHeight;
    }

    async function pollStatus(jobId) {
        // 350 ms feels live without flooding the server. Short videos (Part 1
        // on GPU) finish whole stages in under a second; an 800 ms poll used to
        // miss the entire "Extracting Features" tick.
        while (true) {
            await new Promise(r => setTimeout(r, 350));
            const r = await fetch(`/api/status/${jobId}`);
            if (!r.ok) throw new Error(`Status poll failed: HTTP ${r.status}`);
            const j = await r.json();

            // Map server overall (0..1) into the 30..100% visual range
            const overall = 0.30 + 0.70 * (j.overall || 0);
            progFill.style.width = `${Math.round(overall * 100)}%`;
            progPercent.textContent = fmtPct(overall);
            progMessage.textContent = j.message || j.stage || 'Working…';
            renderPills(j.stage, j.status);

            // If the stage_message carried a BLIP-2 caption, pull it into the feed
            // (server formats as: "Extracting features… frame 120/3600  ·  \u201C ... \u201D")
            const m = (j.message || '').match(/[\u201C"]([^\u201D"]{2,})[\u201D"]/);
            if (m) maybePushCaption(m[1]);

            if (j.status === 'done') return j;
            if (j.status === 'error') throw new Error(j.error || 'Unknown error');
        }
    }

    function renderResults(jobId, final) {
        const res = final.result || {};
        const variant = final.variant || selectedVariant;

        // Set video sources (cache-bust implicit: unique job id)
        vOrig.src = `/api/video/${jobId}/original`;
        vSumm.src = `/api/video/${jobId}/summary`;
        downloadBtn.href = `/api/video/${jobId}/download`;

        // Stats
        const dur = res.duration_seconds || 0;
        const sdur = res.summary_duration_seconds || 0;
        const ratio = dur > 0 ? (sdur / dur) : 0;
        statDur.textContent  = fmtTime(dur);
        statSumm.textContent = fmtTime(sdur);
        statRatio.textContent = ratio > 0 ? `${(ratio * 100).toFixed(1)}%` : '—';
        statShots.textContent = res.num_shots != null ? String(res.num_shots) : '—';

        // Variant badge in results header
        resultVariantBadge.innerHTML = variantBadgeHTML(variant, res.variant_label) +
            (res.feature_label ? `<span class="ui ml-2 text-[12px] text-ink-mute">· ${res.feature_label}</span>` : '');

        // Colour of "AI Summary" title reflects the model used
        if (summaryTitle) {
            summaryTitle.classList.toggle('text-accent',  variant !== 'part2');
            summaryTitle.classList.toggle('text-accent2', variant === 'part2');
            summaryTitle.textContent = variant === 'part2' ? 'Multimodal Summary' : 'AI Summary';
        }

        // Part 2: sample captions gallery
        if (variant === 'part2' && res.sample_captions && res.sample_captions.length > 0) {
            sampleCaptions.innerHTML = '';
            res.sample_captions.forEach(c => {
                const chip = document.createElement('span');
                chip.className = 'mono px-2 py-1 rounded border border-indigo-200 bg-indigo-50 text-indigo-900 text-[12px]';
                chip.textContent = `“${c}”`;
                sampleCaptions.appendChild(chip);
            });
            show(sampleCaptionsWrap);
        } else {
            hide(sampleCaptionsWrap);
        }

        // Part 2: BLIP / CLIP-text features are computed and shipped, but the
        // visualisations stay collapsed until the user clicks the toggle. We
        // stash the data here and render lazily inside the toggle handler.
        if (variant === 'part2' && res.captions_full && res.captions_full.length > 0) {
            vizCachedData = {
                captions: res.captions_full,
                points2d: res.caption_embedding_2d || [],
                selected: res.caption_selected || [],
                shotIds: res.caption_shot_ids || [],
                variance: res.caption_embedding_variance || [0, 0],
            };
            vizRendered = false;
            if (vizPart2Body) vizPart2Body.classList.add('hidden');
            if (vizToggleLabel) vizToggleLabel.textContent = 'Show caption visualisations';
            show(vizPart2Wrap);
        } else {
            vizCachedData = null;
            hide(vizPart2Wrap);
        }

        hide(progSec);
        show(resultsSec);
        resultsSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ---------- Part 2 visualisations ----------
    const STOPWORDS = new Set([
        'a','an','the','and','or','but','of','on','in','at','to','for','with','by','from','as','is','are','was','were',
        'be','been','being','it','its','this','that','these','those','there','their','them','they','he','she','his',
        'her','him','we','our','us','i','me','my','you','your','yours','who','what','which','when','where','why',
        'how','s','t','can','will','just','not','no','very','too','so','than','then','some','any','all','each','few',
        'more','most','other','several','such','only','own','same','up','down','out','over','under','about','into',
        'through','between','also','has','have','had','do','does','did','doing','been','having','while','during'
    ]);

    function renderWordCloud(captions) {
        const counts = new Map();
        for (const cap of captions) {
            const tokens = String(cap || '')
                .toLowerCase()
                .replace(/[^a-z0-9\s'-]/g, ' ')
                .split(/\s+/)
                .filter(Boolean);
            for (const tok of tokens) {
                if (tok.length < 3) continue;
                if (STOPWORDS.has(tok)) continue;
                counts.set(tok, (counts.get(tok) || 0) + 1);
            }
        }
        const sorted = [...counts.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 36);
        if (sorted.length === 0) {
            wordcloudEl.innerHTML = '<div class="ui text-[13px] text-ink-mute">No captions to analyse.</div>';
            return;
        }
        const maxC = sorted[0][1];
        const minC = sorted[sorted.length - 1][1];
        wordcloudEl.innerHTML = '';
        sorted.forEach(([word, count], i) => {
            const span = document.createElement('span');
            span.className = 'wc';
            // Font size: 14 – 42 px, proportional to sqrt(frequency)
            const norm = (Math.sqrt(count) - Math.sqrt(minC)) /
                         (Math.sqrt(maxC) - Math.sqrt(minC) + 1e-6);
            const size = 14 + norm * 28;
            const weight = 400 + Math.round(norm * 300);
            const hue = 220 + (i % 6) * 4; // indigo-ish palette
            const lightness = 30 + Math.round((1 - norm) * 30);
            span.style.fontSize = `${size.toFixed(1)}px`;
            span.style.fontWeight = String(weight);
            span.style.color = `hsl(${hue}, 65%, ${lightness}%)`;
            span.textContent = word;
            span.title = `${word} · ${count}×`;
            wordcloudEl.appendChild(span);
        });
    }

    function renderScatter(points2d, selected, captions, variance) {
        const W = 400, H = 260;
        const PAD = 18;
        if (!points2d || points2d.length < 2) {
            scatterEl.innerHTML = '';
            return;
        }
        // points2d is already normalised to ~[-1, 1] by the backend
        const cx = (x) => PAD + (x + 1) * 0.5 * (W - 2 * PAD);
        const cy = (y) => PAD + (1 - (y + 1) * 0.5) * (H - 2 * PAD);

        let svg = '';
        // Grid lines
        svg += `<line class="grid" x1="${PAD}" y1="${H/2}" x2="${W-PAD}" y2="${H/2}"/>`;
        svg += `<line class="grid" x1="${W/2}" y1="${PAD}" x2="${W/2}" y2="${H-PAD}"/>`;
        // Axes (dashed)
        svg += `<rect x="${PAD-1}" y="${PAD-1}" width="${W-2*PAD+2}" height="${H-2*PAD+2}" fill="none" stroke="#e2e8f0" stroke-width="1"/>`;

        points2d.forEach((p, i) => {
            const [x, y] = p;
            const sel = !!selected[i];
            const fill = sel ? '#1d4ed8' : '#cbd5e1';
            const r = sel ? 5.5 : 3.5;
            const cls = sel ? 'pt selected' : 'pt unselected';
            const cap = captions[i] ? String(captions[i]).replace(/"/g, '\u201D') : '';
            svg += `<circle class="${cls}" cx="${cx(x).toFixed(1)}" cy="${cy(y).toFixed(1)}" r="${r}" fill="${fill}"><title>${escapeHtml(cap)}</title></circle>`;
        });

        // Axis labels
        svg += `<text x="${W-PAD}" y="${H/2 - 5}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="9" fill="#94a3b8">PC1</text>`;
        svg += `<text x="${W/2 + 4}" y="${PAD + 10}" text-anchor="start" font-family="JetBrains Mono, monospace" font-size="9" fill="#94a3b8">PC2</text>`;
        scatterEl.innerHTML = svg;

        if (Array.isArray(variance) && variance.length === 2) {
            const tot = (variance[0] + variance[1]) * 100;
            scatterVariance.textContent = `PC1+PC2 explain ${tot.toFixed(1)}% of variance`;
        } else {
            scatterVariance.textContent = '';
        }
    }

    function renderShotRibbon(captions, shotIds, selected) {
        // Bucket captions by shot id
        const shotMap = new Map();
        for (let i = 0; i < captions.length; i++) {
            const sid = shotIds[i] == null ? 0 : Number(shotIds[i]);
            if (!shotMap.has(sid)) {
                shotMap.set(sid, { shotId: sid, caps: [], selCount: 0, total: 0, firstIdx: i });
            }
            const rec = shotMap.get(sid);
            rec.caps.push(captions[i] || '');
            rec.total += 1;
            if (selected[i]) rec.selCount += 1;
        }
        const shots = [...shotMap.values()].sort((a, b) => a.shotId - b.shotId);
        if (shots.length === 0) {
            shotRibbonEl.innerHTML = '';
            shotRibbonList.innerHTML = '';
            return;
        }

        // Horizontal bar ribbon
        const W = 1000, H = 80;
        const BAR_Y = 24, BAR_H = 34;
        const barW = W / shots.length;
        let svg = '';
        shots.forEach((s, i) => {
            const frac = s.total > 0 ? s.selCount / s.total : 0;
            // Any coverage counts as "selected" (knapsack either picks a shot or not)
            const isSel = frac > 0.05;
            const fill = isSel ? '#1d4ed8' : '#e2e8f0';
            const stroke = isSel ? '#1e40af' : '#cbd5e1';
            svg += `<rect class="shot-tick" x="${(i * barW).toFixed(2)}" y="${BAR_Y}" width="${(barW - 1).toFixed(2)}" height="${BAR_H}" fill="${fill}" stroke="${stroke}"><title>${escapeHtml(firstCap(s.caps))}</title></rect>`;
            if (i === 0 || i === shots.length - 1 || (shots.length > 10 && i % Math.ceil(shots.length / 10) === 0)) {
                svg += `<text class="shot-label" x="${((i + 0.5) * barW).toFixed(2)}" y="${BAR_Y + BAR_H + 12}" text-anchor="middle">shot ${i + 1}</text>`;
            }
        });
        // Legend
        svg += `<text class="shot-label" x="0" y="14">selected into summary</text>`;
        svg += `<rect x="155" y="6" width="10" height="9" fill="#1d4ed8"/>`;
        svg += `<text class="shot-label" x="180" y="14">skipped</text>`;
        svg += `<rect x="218" y="6" width="10" height="9" fill="#e2e8f0" stroke="#cbd5e1"/>`;
        shotRibbonEl.innerHTML = svg;

        // Listing of representative captions
        shotRibbonList.innerHTML = '';
        shots.forEach((s, i) => {
            const row = document.createElement('div');
            row.className = 'flex items-baseline gap-3';
            const dot = `<span class="inline-block w-2 h-2 rounded-full mt-1" style="background:${s.selCount > 0 ? '#1d4ed8' : '#cbd5e1'}"></span>`;
            const idx = `<span class="mono text-[11px] text-ink-mute w-14 shrink-0">shot ${String(i + 1).padStart(2, '0')}</span>`;
            const cap = `<span class="italic">"${escapeHtml(firstCap(s.caps))}"</span>`;
            row.innerHTML = `${dot}${idx}${cap}`;
            shotRibbonList.appendChild(row);
        });
    }

    function firstCap(caps) {
        const first = caps.find(c => c && c.trim().length > 0) || '';
        return first.length > 120 ? first.slice(0, 117) + '…' : first;
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function showError(msg) {
        hide(progSec); hide(resultsSec); show(uploadSec);
        errorMsg.textContent = msg;
        show(errorSec);
    }

    function reset() {
        try { vOrig.pause(); vOrig.removeAttribute('src'); vOrig.load(); } catch {}
        try { vSumm.pause(); vSumm.removeAttribute('src'); vSumm.load(); } catch {}
        fileInput.value = '';
        captionsFeed.innerHTML = '';
        if (wordcloudEl)    wordcloudEl.innerHTML = '';
        if (scatterEl)      scatterEl.innerHTML = '';
        if (shotRibbonEl)   shotRibbonEl.innerHTML = '';
        if (shotRibbonList) shotRibbonList.innerHTML = '';
        if (vizPart2Wrap)   vizPart2Wrap.classList.add('hidden');
        hide(resultsSec); hide(progSec); hide(errorSec);
        show(uploadSec);
        uploadSec.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    resetBtn.addEventListener('click', reset);
    resetErr.addEventListener('click', reset);

    // ---------- BibTeX copy ----------
    copyBibtex.addEventListener('click', async () => {
        const visible = document.querySelector('.bibtex-body:not(.hidden)');
        const text = (visible || document.querySelector('.bibtex-body')).innerText;
        try {
            await navigator.clipboard.writeText(text);
            copyFeedback.textContent = 'Copied ✓';
        } catch {
            copyFeedback.textContent = 'Copy failed — select manually';
        }
        setTimeout(() => (copyFeedback.textContent = ''), 2000);
    });

    // Initialize variant visuals
    setVariant('part1');
})();
