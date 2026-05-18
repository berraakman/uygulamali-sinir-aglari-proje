/* ══════════════════════════════════════════════════════════════
   DepthVision – JavaScript Controller
   Drag & drop, upload, prediction, and UI interactions
   ══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
    // ─── DOM Elements ───────────────────────────────────────────
    const uploadSection = document.getElementById("uploadSection");
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");

    const loadingSection = document.getElementById("loadingSection");
    const progressBar = document.getElementById("progressBar");

    const resultsSection = document.getElementById("resultsSection");
    const comparisonImage = document.getElementById("comparisonImage");
    const depthImage = document.getElementById("depthImage");

    const tabComparison = document.getElementById("tabComparison");
    const tabDepth = document.getElementById("tabDepth");
    const panelComparison = document.getElementById("panelComparison");
    const panelDepth = document.getElementById("panelDepth");

    const statMin = document.getElementById("statMin");
    const statMax = document.getElementById("statMax");
    const statMean = document.getElementById("statMean");
    const statStd = document.getElementById("statStd");

    const newImageBtn = document.getElementById("newImageBtn");
    const downloadBtn = document.getElementById("downloadBtn");

    // ─── Background Particles ───────────────────────────────────
    const bgParticles = document.getElementById("bgParticles");
    function createParticles() {
        const colors = [
            "rgba(167, 139, 250, 0.3)",
            "rgba(99, 102, 241, 0.25)",
            "rgba(139, 92, 246, 0.2)",
            "rgba(236, 72, 153, 0.15)",
        ];

        for (let i = 0; i < 25; i++) {
            const particle = document.createElement("div");
            particle.className = "particle";
            const size = Math.random() * 4 + 2;
            particle.style.width = size + "px";
            particle.style.height = size + "px";
            particle.style.left = Math.random() * 100 + "%";
            particle.style.background = colors[Math.floor(Math.random() * colors.length)];
            particle.style.animationDuration = (Math.random() * 15 + 10) + "s";
            particle.style.animationDelay = (Math.random() * 10) + "s";
            bgParticles.appendChild(particle);
        }
    }
    createParticles();

    // ─── Drag & Drop ────────────────────────────────────────────
    let dragCounter = 0;

    uploadZone.addEventListener("dragenter", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragCounter++;
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragCounter--;
        if (dragCounter === 0) {
            uploadZone.classList.remove("drag-over");
        }
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragCounter = 0;
        uploadZone.classList.remove("drag-over");

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // ─── Click to Upload ────────────────────────────────────────
    uploadZone.addEventListener("click", (e) => {
        if (e.target === browseBtn || browseBtn.contains(e.target)) return;
        fileInput.click();
    });

    browseBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // ─── Tab Switching ──────────────────────────────────────────
    tabComparison.addEventListener("click", () => switchTab("comparison"));
    tabDepth.addEventListener("click", () => switchTab("depth"));

    function switchTab(tab) {
        tabComparison.classList.toggle("active", tab === "comparison");
        tabDepth.classList.toggle("active", tab === "depth");
        panelComparison.classList.toggle("active", tab === "comparison");
        panelDepth.classList.toggle("active", tab === "depth");
    }

    // ─── New Image Button ───────────────────────────────────────
    newImageBtn.addEventListener("click", () => {
        resultsSection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
        fileInput.value = "";
    });

    // ─── Download Button ────────────────────────────────────────
    downloadBtn.addEventListener("click", () => {
        const activeImg = panelComparison.classList.contains("active")
            ? comparisonImage
            : depthImage;

        const link = document.createElement("a");
        link.href = activeImg.src;
        link.download = "depth_result.png";
        link.click();
    });

    // ─── Handle File Upload ─────────────────────────────────────
    function handleFile(file) {
        // Validate
        const allowed = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp", "image/tiff"];
        if (!allowed.includes(file.type) && !file.name.match(/\.(png|jpe?g|bmp|webp|tiff?)$/i)) {
            showError("Desteklenmeyen dosya formatı! PNG, JPG, JPEG, WebP, BMP veya TIFF yükleyin.");
            return;
        }

        // Show loading
        uploadSection.classList.add("hidden");
        resultsSection.classList.add("hidden");
        loadingSection.classList.remove("hidden");

        // Animate progress
        animateProgress();

        // Upload
        const formData = new FormData();
        formData.append("image", file);

        fetch("/predict", {
            method: "POST",
            body: formData,
        })
        .then((res) => {
            if (!res.ok) return res.json().then((d) => { throw new Error(d.error || "Sunucu hatası"); });
            return res.json();
        })
        .then((data) => {
            // Set progress to 100%
            progressBar.style.width = "100%";

            setTimeout(() => {
                showResults(data);
            }, 400);
        })
        .catch((err) => {
            loadingSection.classList.add("hidden");
            uploadSection.classList.remove("hidden");
            showError(err.message);
        });
    }

    // ─── Progress Animation ─────────────────────────────────────
    let progressInterval;
    function animateProgress() {
        let progress = 0;
        progressBar.style.width = "0%";

        clearInterval(progressInterval);
        progressInterval = setInterval(() => {
            if (progress < 85) {
                progress += Math.random() * 8 + 2;
                if (progress > 85) progress = 85;
                progressBar.style.width = progress + "%";
            }
        }, 200);
    }

    // ─── Show Results ───────────────────────────────────────────
    function showResults(data) {
        clearInterval(progressInterval);

        // Set images
        comparisonImage.src = "data:image/png;base64," + data.comparison_image;
        depthImage.src = "data:image/png;base64," + data.depth_image;

        // Set stats
        statMin.textContent = data.stats.min_depth;
        statMax.textContent = data.stats.max_depth;
        statMean.textContent = data.stats.mean_depth;
        statStd.textContent = data.stats.std_depth;

        // Switch views
        loadingSection.classList.add("hidden");
        resultsSection.classList.remove("hidden");
        switchTab("comparison");

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ─── Error Toast ────────────────────────────────────────────
    function showError(message) {
        // Create toast
        const toast = document.createElement("div");
        toast.style.cssText = `
            position: fixed;
            top: 24px;
            right: 24px;
            z-index: 9999;
            padding: 16px 24px;
            border-radius: 12px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            font-family: var(--font-main);
            font-size: 0.9rem;
            font-weight: 500;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.3s ease-out;
            max-width: 400px;
        `;
        toast.textContent = "⚠️ " + message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transition = "opacity 0.3s";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});
