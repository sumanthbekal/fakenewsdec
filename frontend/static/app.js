document.addEventListener('DOMContentLoaded', () => {
  const articleInput = document.getElementById('article-input');
  const wordCountDisplay = document.getElementById('word-count');
  const wordCountBadge = document.getElementById('word-count-badge');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnClear = document.getElementById('btn-clear');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  const errorAlert = document.getElementById('error-alert');
  const errorMessage = document.getElementById('error-message');
  const resultCard = document.getElementById('result-card');
  const resultTag = document.getElementById('result-tag');
  const statConfidence = document.getElementById('stat-confidence');
  const statWords = document.getElementById('stat-words');
  const probFakeText = document.getElementById('prob-fake-text');
  const probRealText = document.getElementById('prob-real-text');
  const barFake = document.getElementById('bar-fake');
  const barReal = document.getElementById('bar-real');

  // Sample articles matching prompt requirements
  const sampleArticles = {
    real1: "The Oakridge City Council unanimously approved a comprehensive $14.5 million municipal road repair and resurfacing budget yesterday evening. City Transportation Director Mark Henderson announced that funding has been allocated from the state infrastructure grant and local reserve capital. The phased construction project will target major arterial roadways including Elmwood Boulevard, 5th Avenue, and North Commerce Way, beginning early this spring with a projected completion date set for November 2027.",
    real2: "The State University Academic Board today officially announced the inauguration of its 2026-2027 Presidential Merit Scholarships for incoming undergraduate and graduate students. According to university provost Dr. Susan Vance, eligible applicants must possess a cumulative 3.8 GPA and submit verified community service portfolios before the March 15th priority deadline. Selected recipients will receive up to 100% tuition coverage along with annual research stipends.",
    fake1: "SHOCKING SECRET REVEALED: Big Pharma scientists and elite corrupt doctors are desperately hiding a 100% natural miracle cancer cure discovered in tropical rain forests! Whistleblowers report government agencies are deleting all scientific files and threatening researchers with immediate arrest. Read this truth before the global elites censor this website and delete this video forever, share with everyone now!",
    fake2: "CONFIDENTIAL LEAK FROM ANONYMOUS WALL STREET INSIDER: All major national banks and federal reserve institutions are secretly planning to shut down completely for thirty consecutive days starting next week! Millions of civilian retirement accounts and hard-earned savings deposits will permanently vanish under emergency financial protocol martial law. Massive government cover-up exposed by top insiders."
  };

  // Live word counter
  function updateWordCount() {
    const text = articleInput.value.trim();
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
    wordCountDisplay.textContent = words;
    if (words >= 10) {
      wordCountBadge.classList.add('valid');
    } else {
      wordCountBadge.classList.remove('valid');
    }
    return words;
  }

  articleInput.addEventListener('input', () => {
    updateWordCount();
    hideError();
  });

  // Example buttons click
  document.querySelectorAll('.btn-example').forEach(button => {
    button.addEventListener('click', () => {
      const sampleKey = button.getAttribute('data-sample');
      if (sampleArticles[sampleKey]) {
        articleInput.value = sampleArticles[sampleKey];
        updateWordCount();
        hideError();
        // Smooth scroll to textarea
        articleInput.focus();
      }
    });
  });

  // Clear button
  btnClear.addEventListener('click', () => {
    articleInput.value = '';
    updateWordCount();
    hideError();
    hideResult();
    articleInput.focus();
  });

  // Show/Hide error
  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.style.display = 'flex';
  }

  function hideError() {
    errorAlert.style.display = 'none';
    errorMessage.textContent = '';
  }

  // Show/Hide results
  function hideResult() {
    resultCard.style.display = 'none';
    resultCard.className = 'card result-card';
  }

  function setLoading(isLoading) {
    if (isLoading) {
      btnAnalyze.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Analyzing Article...';
    } else {
      btnAnalyze.disabled = false;
      btnSpinner.style.display = 'none';
      btnText.textContent = 'Analyze Article';
    }
  }

  // Submit and analyze
  btnAnalyze.addEventListener('click', async () => {
    const rawText = articleInput.value.trim();
    const words = updateWordCount();

    if (!rawText) {
      showError('Please enter or paste news article text before analyzing.');
      return;
    }

    if (words < 10) {
      showError(`The article contains only ${words} word(s). Please enter at least 10 words for accurate linguistic analysis.`);
      return;
    }

    hideError();
    hideResult();
    setLoading(true);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: rawText })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Prediction request failed.');
      }

      displayResult(data);
    } catch (err) {
      showError(err.message || 'An error occurred while connecting to the detector API.');
    } finally {
      setLoading(false);
    }
  });

  function displayResult(data) {
    const isFake = data.label === 'FAKE';

    resultCard.style.display = 'block';
    resultCard.className = `card result-card ${isFake ? 'fake' : 'real'}`;

    resultTag.className = `result-tag ${isFake ? 'fake' : 'real'}`;
    resultTag.innerHTML = isFake 
      ? `<span>⚠️ Predicted FAKE</span>` 
      : `<span>✅ Predicted REAL</span>`;

    const confPct = (data.confidence * 100).toFixed(1);
    statConfidence.textContent = `${confPct}%`;
    statWords.textContent = data.words;

    const fakePct = (data.probabilities.FAKE * 100).toFixed(1);
    const realPct = (data.probabilities.REAL * 100).toFixed(1);

    probFakeText.textContent = `${fakePct}%`;
    probRealText.textContent = `${realPct}%`;

    // Trigger animation
    setTimeout(() => {
      barFake.style.width = `${fakePct}%`;
      barReal.style.width = `${realPct}%`;
    }, 50);

    // Scroll smoothly to results
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Initial fetch for model health / metrics
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        if (data.metrics) {
          const accEl = document.getElementById('metric-accuracy');
          const f1El = document.getElementById('metric-f1');
          const aucEl = document.getElementById('metric-auc');
          const precEl = document.getElementById('metric-precision');
          if (accEl) accEl.textContent = `${(data.metrics.accuracy * 100).toFixed(1)}%`;
          if (f1El) f1El.textContent = `${(data.metrics.f1 * 100).toFixed(1)}%`;
          if (aucEl) aucEl.textContent = `${(data.metrics.roc_auc * 100).toFixed(1)}%`;
          if (precEl) precEl.textContent = `${(data.metrics.precision * 100).toFixed(1)}%`;
        }
      }
    } catch (e) {
      console.log('Health check note:', e);
    }
  }

  checkHealth();
  updateWordCount();
});
