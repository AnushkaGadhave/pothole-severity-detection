/**
 * ==============================================================================
 * CRACKSIGHT — FRONTEND LOGIC & INTERACTION ENGINE
 * Theme: Multicolour Glassmorphism × Motion Design × Computer Vision
 *
 * Features:
 * - Coordinated Staggered Scroll Reveals (IntersectionObserver)
 * - Dynamic Scrolled Navbar & Active Section Tracking
 * - Interactive Multi-Mode Product-Preview Workbench
 * - Scan Beam Lifecycle (Pauses when offscreen, reduced-motion compliant)
 * - Animated Real Dataset Distribution Chart (Pothole440, N=434)
 * - Interactive Clickable Workflow Stages with Dynamic Detail Panel
 * - Smooth Back-to-Top Navigation
 * ==============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveals();
  initNavbarBehavior();
  initProductPreviewWorkbench();
  initDatasetDistributionChart();
  initInteractiveWorkflow();
  initBackToTop();
});

/**
 * 1. Coordinated Staggered Scroll Reveals
 */
function initScrollReveals() {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const revealElements = document.querySelectorAll('.reveal-on-scroll');

  if (prefersReducedMotion || !('IntersectionObserver' in window)) {
    revealElements.forEach(el => el.classList.add('is-revealed'));
    return;
  }

  const observerOptions = {
    root: null,
    rootMargin: '0px 0px -40px 0px',
    threshold: 0.12
  };

  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-revealed');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  revealElements.forEach(el => revealObserver.observe(el));
}

/**
 * 2. Sticky Navbar, Scrolled State & Mobile Drawer Control
 */
function initNavbarBehavior() {
  const navbar = document.getElementById('navbar');
  const mobileToggle = document.getElementById('mobileNavToggle');
  const mobileDrawer = document.getElementById('mobileDrawer');
  const mobileLinks = document.querySelectorAll('.mobile-nav-link');
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('section[id], header[id]');

  // Scrolled styling & Active link tracking
  window.addEventListener('scroll', () => {
    const scrollPos = window.scrollY;

    // Toggle .scrolled shadow/translucency class
    if (navbar) {
      if (scrollPos > 30) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    }

    // Active link highlighting
    let currentId = '';
    sections.forEach(sec => {
      const top = sec.offsetTop - 140;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute('id');
      }
    });

    if (currentId) {
      navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentId}`) {
          link.classList.add('active');
        }
      });
    }
  }, { passive: true });

  // Mobile Drawer Toggle (only active on mobile viewports)
  if (mobileToggle && mobileDrawer) {
    mobileToggle.addEventListener('click', () => {
      const isExpanded = mobileToggle.getAttribute('aria-expanded') === 'true';
      mobileToggle.setAttribute('aria-expanded', !isExpanded);
      mobileDrawer.classList.toggle('open');
    });

    mobileLinks.forEach(link => {
      link.addEventListener('click', () => {
        mobileDrawer.classList.remove('open');
        mobileToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }
}

/**
 * 3. Interactive Product-Preview Workbench & Scan Beam Control
 */
function initProductPreviewWorkbench() {
  const modeButtons = document.querySelectorAll('.mode-btn');
  const workbenchCanvas = document.querySelector('.workbench-canvas');
  const panelSource = document.getElementById('panelSourceView');
  const panelMask = document.getElementById('panelSegmentationView');
  const geometrySvg = document.getElementById('geometrySvgElem');
  const scanBeam = document.getElementById('scanBeamElem');
  const previewSection = document.getElementById('preview');

  // Pause scan line animation when preview section is not visible
  if (scanBeam && previewSection && 'IntersectionObserver' in window) {
    const scanObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          scanBeam.style.animationPlayState = 'running';
        } else {
          scanBeam.style.animationPlayState = 'paused';
        }
      });
    }, { threshold: 0.1 });
    scanObserver.observe(previewSection);
  }

  if (!modeButtons.length) return;

  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      modeButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const mode = btn.getAttribute('data-mode');

      // Adjust Workbench Visual Panels smoothly
      if (mode === 'rgb') {
        if (panelSource) panelSource.style.display = 'flex';
        if (panelMask) panelMask.style.display = 'none';
        if (scanBeam) scanBeam.style.display = 'block';
        if (workbenchCanvas) workbenchCanvas.classList.add('single-panel-view');
      } else if (mode === 'mask') {
        if (panelSource) panelSource.style.display = 'none';
        if (panelMask) panelMask.style.display = 'flex';
        if (geometrySvg) geometrySvg.style.display = 'none';
        if (workbenchCanvas) workbenchCanvas.classList.add('single-panel-view');
      } else if (mode === 'geometry') {
        if (panelSource) panelSource.style.display = 'none';
        if (panelMask) panelMask.style.display = 'flex';
        if (geometrySvg) geometrySvg.style.display = 'block';
        if (workbenchCanvas) workbenchCanvas.classList.add('single-panel-view');
      } else {
        // Composite (Default)
        if (panelSource) panelSource.style.display = 'flex';
        if (panelMask) panelMask.style.display = 'flex';
        if (geometrySvg) geometrySvg.style.display = 'block';
        if (scanBeam) scanBeam.style.display = 'block';
        if (workbenchCanvas) workbenchCanvas.classList.remove('single-panel-view');
      }
    });
  });
}

/**
 * 4. Animated Real Dataset Distribution Chart
 * Data: models/model_metadata.json (N = 434)
 */
function initDatasetDistributionChart() {
  const chartWrapper = document.getElementById('datasetDistributionChart');
  if (!chartWrapper) return;

  const chartFills = chartWrapper.querySelectorAll('.chart-fill');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const triggerFill = () => {
    chartFills.forEach(fill => {
      const targetWidth = fill.getAttribute('data-target-width');
      fill.style.width = targetWidth || '0%';
    });
  };

  if (prefersReducedMotion || !('IntersectionObserver' in window)) {
    triggerFill();
    return;
  }

  const chartObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        setTimeout(triggerFill, 250);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.25 });

  chartObserver.observe(chartWrapper);
}

/**
 * 5. Interactive Clickable Workflow Stages with Dynamic Detail Panel
 */
function initInteractiveWorkflow() {
  const stageCards = document.querySelectorAll('.stage-step-card');
  const detailTag = document.getElementById('workflowDetailTag');
  const detailTitle = document.getElementById('workflowDetailTitle');
  const detailCopy = document.getElementById('workflowDetailCopy');
  const detailSpecs = document.getElementById('workflowDetailSpecs');

  if (!stageCards.length) return;

  const stageData = [
    {
      stage: 1,
      tag: 'STAGE 01 OF 04 · INPUT INGESTION',
      title: 'Provide image or required data',
      copy: 'RGB imagery is used for segmentation, while the feature-based severity pipeline may require corresponding depth-map-derived information.',
      specs: [
        { k: 'Modality:', v: 'Standard 3-Channel RGB (H×W×3)' },
        { k: 'Spatial Scaling:', v: '0.25 Scale Normalized Matrix' },
        { k: 'Sensor Telemetry:', v: 'Optional Uint8 [0–190] Grayscale Depth Array' }
      ]
    },
    {
      stage: 2,
      tag: 'STAGE 02 OF 04 · 2D REGION PREDICTION',
      title: 'Segment the pothole region',
      copy: 'The experimental RGB model predicts a binary mask of the likely pothole region using multi-scale color contrast, luminance gradient filters, and morphological closing.',
      specs: [
        { k: 'Detection Paradigm:', v: 'Multi-scale Feature Classifier' },
        { k: 'Post-Processing:', v: '3×3 Morphological Binary Closing & Hole-Filling' },
        { k: 'Evaluation Protocol:', v: 'Strict Group-Disjoint 5-Fold Cross Validation' }
      ]
    },
    {
      stage: 3,
      tag: 'STAGE 03 OF 04 · GEOMETRIC EXTRACTION',
      title: 'Extract image-derived features',
      copy: 'The project examines 14 planar geometric footprint descriptors (area, perimeter, circularity, aspect ratio, solidity) computed strictly in uncalibrated image pixels.',
      specs: [
        { k: '2D Planar Features:', v: '14 Morphometric Descriptors Extracted' },
        { k: 'Sensor Depth Features:', v: 'Relative Depression Depth & Basin Volume Proxy' },
        { k: 'Coordinate Bounds:', v: 'Strictly Image Pixel Space (Non-Metric)' }
      ]
    },
    {
      stage: 4,
      tag: 'STAGE 04 OF 04 · PROVISIONAL ASSESSMENT',
      title: 'Review the preliminary assessment',
      copy: 'The formula-based pipeline evaluates the relative severity index (RPHI) to assign provisional relative tiers (Low, Medium, High) for comparative research.',
      specs: [
        { k: 'Provisional Tiers:', v: 'Low (32.5%), Medium (32.9%), High (34.6%)' },
        { k: 'Percentile Split:', v: 'Tertile Quantiles (33% / 66% fitted on train sets)' },
        { k: 'Operational Bound:', v: 'Research Prototype (Not certified municipal rating)' }
      ]
    }
  ];

  let activeIndex = 0;
  let userInteracted = false;
  let cycleTimer = null;

  const updateStageView = (index) => {
    activeIndex = index;
    const data = stageData[index];

    // Highlight active card
    stageCards.forEach((card, idx) => {
      if (idx === index) {
        card.classList.add('active-flow');
        card.setAttribute('aria-selected', 'true');
      } else {
        card.classList.remove('active-flow');
        card.setAttribute('aria-selected', 'false');
      }
    });

    // Update detail panel content with smooth fade
    if (detailTag) detailTag.textContent = data.tag;
    if (detailTitle) detailTitle.textContent = data.title;
    if (detailCopy) detailCopy.textContent = data.copy;
    if (detailSpecs) {
      detailSpecs.innerHTML = data.specs
        .map(s => `<div class="spec-item"><span class="spec-k">${s.k}</span> <span class="spec-v">${s.v}</span></div>`)
        .join('');
    }
  };

  // Add click & keyboard listeners to each stage card
  stageCards.forEach((card, idx) => {
    card.addEventListener('click', () => {
      userInteracted = true;
      if (cycleTimer) clearInterval(cycleTimer);
      updateStageView(idx);
    });

    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        userInteracted = true;
        if (cycleTimer) clearInterval(cycleTimer);
        updateStageView(idx);
      }
    });
  });

  // Ambient automatic sequence for discovery (disabled if reduced-motion preferred or user has clicked)
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!prefersReducedMotion) {
    cycleTimer = setInterval(() => {
      if (userInteracted) {
        clearInterval(cycleTimer);
        return;
      }
      activeIndex = (activeIndex + 1) % stageCards.length;
      updateStageView(activeIndex);
    }, 3200);
  }
}

/**
 * 6. Smooth Back-to-Top Navigation
 */
function initBackToTop() {
  const backBtn = document.getElementById('backToTopBtn');
  if (!backBtn) return;

  backBtn.addEventListener('click', (e) => {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
}
