"use strict";

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("upload-form");
    const input = document.getElementById("image-input");
    const dropZone = document.getElementById("drop-zone");
    const browseButton = document.getElementById("browse-button");

    const previewState = document.getElementById("preview-state");
    const previewImage = document.getElementById("image-preview");
    const fileName = document.getElementById("file-name");
    const removeButton = document.getElementById("remove-image");

    const analyzeButton = document.getElementById("analyze-button");
    const buttonLabel = document.getElementById("button-label");
    const spinner = document.getElementById("button-spinner");
    const arrow = document.getElementById("button-arrow");
    const errorMessage = document.getElementById("error-message");

    const results = document.getElementById("prediction-results");
    const resultImage = document.getElementById("result-image");
    const resultFilename = document.getElementById("result-filename");
    const predictedClass = document.getElementById("predicted-class");
    const analyzeAnother = document.getElementById("analyze-another");

    const modelReady = form?.dataset.modelReady === "true";

    const allowedTypes = ["image/jpeg", "image/png"];
    const maxFileSize = 10 * 1024 * 1024;

    let selectedFile = null;
    let previewUrl = null;
    let isAnalyzing = false;

    function showError(message) {
        if (!errorMessage) return;
        errorMessage.textContent = message;
        errorMessage.hidden = false;
    }

    function clearError() {
        if (!errorMessage) return;
        errorMessage.textContent = "";
        errorMessage.hidden = true;
    }

    function setLoading(loading) {
        isAnalyzing = loading;

        if (analyzeButton) {
            analyzeButton.disabled =
                loading || !selectedFile || !modelReady;
        }

        if (buttonLabel) {
            buttonLabel.textContent = loading
                ? "Analyzing Image..."
                : "Predict Cell Type";
        }

        if (spinner) spinner.hidden = !loading;
        if (arrow) arrow.hidden = loading;
    }

    function clearPreviewUrl() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    function selectImage(file) {
        if (isAnalyzing || !file) return;

        clearError();

        if (!allowedTypes.includes(file.type)) {
            showError("Please select a JPG, JPEG, or PNG image.");
            return;
        }

        if (file.size === 0 || file.size > maxFileSize) {
            showError("Please select a valid image smaller than 10 MB.");
            return;
        }

        clearPreviewUrl();

        selectedFile = file;
        previewUrl = URL.createObjectURL(file);

        if (previewImage) previewImage.src = previewUrl;
        if (fileName) fileName.textContent = file.name;
        if (previewState) previewState.hidden = false;
        if (results) results.hidden = true;

        if (analyzeButton) {
            analyzeButton.disabled = !modelReady;
        }
    }

    if (browseButton && input) {
        browseButton.addEventListener("click", function () {
            input.click();
        });
    }

    if (input) {
        input.addEventListener("change", function () {
            if (input.files.length) {
                selectImage(input.files[0]);
            }
        });
    }

    if (dropZone && input) {
        dropZone.addEventListener("click", function (event) {
            if (event.target.closest("button")) return;
            input.click();
        });

        ["dragenter", "dragover"].forEach(function (eventName) {
            dropZone.addEventListener(eventName, function (event) {
                event.preventDefault();
                dropZone.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(function (eventName) {
            dropZone.addEventListener(eventName, function (event) {
                event.preventDefault();
                dropZone.classList.remove("drag-over");
            });
        });

        dropZone.addEventListener("drop", function (event) {
            const files = event.dataTransfer.files;
            if (files.length) selectImage(files[0]);
        });
    }

    function resetUpload() {
        if (isAnalyzing) return;

        selectedFile = null;

        if (input) input.value = "";
        if (previewImage) previewImage.removeAttribute("src");
        if (previewState) previewState.hidden = true;
        if (fileName) fileName.textContent = "";
        if (results) results.hidden = true;

        clearPreviewUrl();
        clearError();
        setLoading(false);
    }

    if (removeButton) {
        removeButton.addEventListener("click", resetUpload);
    }

    if (form) {
        form.addEventListener("submit", async function (event) {
            event.preventDefault();
            clearError();

            if (isAnalyzing) return;

            if (!selectedFile) {
                showError("Please select an image first.");
                return;
            }

            if (!modelReady) {
                showError("The prediction model is currently unavailable.");
                return;
            }

            const formData = new FormData();
            formData.append("file", selectedFile);

            setLoading(true);

            try {
                const response = await fetch(form.action, {
                    method: "POST",
                    body: formData,
                    headers: {
                        "Accept": "application/json",
                        "X-Requested-With": "XMLHttpRequest"
                    }
                });

                const data = await response.json();

                if (!response.ok || data.status !== "success") {
                    throw new Error(
                        data.error || "Unable to analyze this image."
                    );
                }

                if (predictedClass) {
                    predictedClass.textContent = data.prediction;
                }

                if (resultImage && data.image_url) {
                    resultImage.src = data.image_url;
                }

                if (resultFilename) {
                    resultFilename.textContent =
                        data.filename || selectedFile.name;
                }

                if (results) {
                    results.hidden = false;
                    results.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }

                clearPreviewUrl();

            } catch (error) {
                console.error("Prediction error:", error);
                showError(error.message || "Something went wrong.");
            } finally {
                setLoading(false);
            }
        });
    }

    if (analyzeAnother) {
        analyzeAnother.addEventListener("click", function () {
            resetUpload();

            if (input) input.click();
        });
    }

    function updateActiveLink() {
        document.querySelectorAll(".nav-link").forEach(function (link) {
            const target = link.getAttribute("href");
            link.classList.toggle(
                "active",
                target === "#" + (location.hash.slice(1) || "dashboard")
            );
        });
    }

    document.querySelectorAll(".nav-link").forEach(function (link) {
        link.addEventListener("click", function () {
            setTimeout(updateActiveLink, 0);
        });
    });

    window.addEventListener("hashchange", updateActiveLink);

    if (results) results.hidden = true;
    if (analyzeButton) analyzeButton.disabled = !modelReady;

    updateActiveLink();
});