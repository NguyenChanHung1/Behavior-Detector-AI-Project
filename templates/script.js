const video = document.getElementById("video");
const startButton = document.getElementById("startBtn");

async function startCamera(s) {
    if(navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({video: true})
        .then(function(s) {
            video.srcObject = s;
            startButton.remove();
        });
    }
    else{
        console.log("No");
    }
}

async function hideBoundingBoxButton() {
    var hideBox = document.createElement("button");

    hideBox.innerText = "Hide Bounding Box";

    hideBox.style.backgroundColor = "red";
    hideBox.style.color = "white";
    hideBox.style.fontSize = "16px";
    hideBox.style.fontFamily = "Franklin Gothic Medium";
    hideBox.style.borderRadius = "20px";
    hideBox.style.padding = "15px 50px 15px 50px";
    hideBox.style.marginTop = "20px";
    hideBox.style.marginLeft = "500px";

    hideBox.addEventListener("click", function() {
        boundingBoxButton();
        hideBox.remove();
    })

    document.getElementById("button-container").appendChild(hideBox);
}

async function boundingBoxButton() {
    var boundingBox = document.createElement("button");

    boundingBox.innerText = "Show Bounding Box";

    boundingBox.style.backgroundColor = "red";
    boundingBox.style.color = "white";
    boundingBox.style.fontSize = "16px";
    boundingBox.style.fontFamily = "Franklin Gothic Medium";
    boundingBox.style.borderRadius = "20px";
    boundingBox.style.padding = "15px 50px 15px 50px";
    boundingBox.style.marginTop = "20px";
    boundingBox.style.marginLeft = "500px";

    boundingBox.addEventListener("click", function() {
        hideBoundingBoxButton();
        boundingBox.remove();
    })

    document.getElementById("button-container").appendChild(boundingBox);
}

startButton.addEventListener('click', startCamera);
startButton.addEventListener('click', boundingBoxButton);

fetch('api/')
    .then(response => response.json())
    .then(data => console.log(data.message))
    .catch(error => console.error(error));