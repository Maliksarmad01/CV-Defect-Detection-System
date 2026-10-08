const imageInput =
    document.getElementById("image-input");

const previewContainer =
    document.getElementById("preview-container");

const imagePreview =
    document.getElementById("image-preview");

const removeImageButton =
    document.getElementById("remove-image");

const analyzeButton =
    document.getElementById("analyze-button");

const errorMessage =
    document.getElementById("error-message");

const loading =
    document.getElementById("loading");

const emptyResult =
    document.getElementById("empty-result");

const resultContent =
    document.getElementById("result-content");

const predictionBadge =
    document.getElementById("prediction-badge");

const confidence =
    document.getElementById("confidence");

const confidenceBar =
    document.getElementById("confidence-bar");

const normalProbability =
    document.getElementById("normal-probability");

const defectiveProbability =
    document.getElementById("defective-probability");

const device =
    document.getElementById("device");


/* =========================================================
   IMAGE SELECTION
========================================================= */

imageInput.addEventListener(
    "change",
    function () {

        const file = this.files[0];

        if (!file) {
            return;
        }


        if (!file.type.startsWith("image/")) {

            showError(
                "Please select a valid image file."
            );

            return;
        }


        const imageURL =
            URL.createObjectURL(file);


        imagePreview.src =
            imageURL;


        previewContainer.style.display =
            "block";


        analyzeButton.disabled =
            false;


        clearError();


        emptyResult.style.display =
            "flex";


        resultContent.style.display =
            "none";
    }
);


/* =========================================================
   REMOVE IMAGE
========================================================= */

removeImageButton.addEventListener(
    "click",
    function () {

        imageInput.value = "";

        imagePreview.src = "";

        previewContainer.style.display =
            "none";

        analyzeButton.disabled =
            true;

        clearError();
    }
);


/* =========================================================
   ANALYZE
========================================================= */

analyzeButton.addEventListener(
    "click",
    async function () {

        const file =
            imageInput.files[0];


        if (!file) {

            showError(
                "Please select an image first."
            );

            return;
        }


        clearError();


        analyzeButton.disabled =
            true;


        loading.style.display =
            "block";


        emptyResult.style.display =
            "none";


        resultContent.style.display =
            "none";


        const formData =
            new FormData();


        formData.append(
            "file",
            file
        );


        try {

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const contentType =
                response.headers.get(
                    "content-type"
                ) || "";


            if (!contentType.includes(
                "application/json"
            )) {

                throw new Error(
                    "The server returned an unexpected response."
                );
            }


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prediction failed."
                );
            }


            displayResult(data);

        }

        catch (error) {

            showError(
                error.message
            );

        }

        finally {

            loading.style.display =
                "none";


            analyzeButton.disabled =
                false;
        }

    }
);


/* =========================================================
   DISPLAY RESULT
========================================================= */

function displayResult(data) {

    const prediction =
        data.prediction;


    const confidenceValue =
        Number(data.confidence);


    const normalValue =
        Number(data.normal_probability);


    const defectiveValue =
        Number(data.defective_probability) ;


    predictionBadge.textContent =
        prediction;


    predictionBadge.classList.remove(
        "normal"
    );


    if (
        prediction.toLowerCase() ===
        "normal"
    ) {

        predictionBadge.classList.add(
            "normal"
        );

    }


    confidence.textContent =
        confidenceValue.toFixed(2) + "%";


    confidenceBar.style.width =
        confidenceValue + "%";


    normalProbability.textContent =
        normalValue.toFixed(2) + "%";


    defectiveProbability.textContent =
        defectiveValue.toFixed(2) + "%";


    device.textContent =
        data.device || "CPU";


    emptyResult.style.display =
        "none";


    resultContent.style.display =
        "block";
}


/* =========================================================
   ERROR
========================================================= */

function showError(message) {

    errorMessage.textContent =
        message;

    errorMessage.style.display =
        "block";
}


function clearError() {

    errorMessage.textContent =
        "";

    errorMessage.style.display =
        "none";
}