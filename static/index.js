const gif = document.getElementById("gif");
const button = document.getElementById("button");
const output = document.getElementById("output");
const thumbsup = document.getElementById("thumbsup");
const thumbsdown = document.getElementById("thumbsdown");
let upcount = 0;
let downcount = 0;
let liked = null;
let translation_id = null;

async function translateText() {
    const englishText = document.getElementById("inputBox").value;
    output.textContent = "Translating...";
    gif.style.display="none";

    thumbsup.textContent=""
    thumbsdown.textContent=""

    thumbsup.disabled=false;
    thumbsdown.disabled=false;

    button.disabled=true;

    const response = await fetch('/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }, 
        body: JSON.stringify({"text": englishText})
    })

    if (!response.ok) {
        output.textContent = "Error translating.";
        button.disabled = false;
        return;
    }

    const data = await response.json();
    const translatedSentence = data['translated_text'];
    translation_id = data['translation_id'];
    output.textContent = translatedSentence;

    gif.src = "/static/right.gif";
    gif.style.display="block";

    thumbsup.textContent=upcount == 0? "👍" : `👍 ${upcount}`;
    thumbsdown.textContent=downcount == 0? "👎" : `👎 ${downcount}`;

    thumbsup.style.display="inline-block";
    thumbsdown.style.display="inline-block";

    button.disabled=false;
}

button.addEventListener("click", translateText);

thumbsup.addEventListener("click", async() => {
    upcount += 1;
    thumbsup.textContent = `👍 ${upcount}`;
    thumbsup.disabled = true;
    thumbsdown.disabled = true;
    liked = true;
    const response = await fetch('/feedback', {
        method: 'POST', 
        headers: {
            'Content-Type': 'application/json'
        }, 
        body: JSON.stringify({"translation_id": translation_id, "liked": liked})
    });

    if (!response.ok) {
        console.error("Error sending feedback");
    }
});

thumbsdown.addEventListener("click", async() => {
    downcount += 1;
    thumbsdown.textContent = `👎 ${downcount}`;
    thumbsup.disabled = true;
    thumbsdown.disabled = true;
    liked = false;
    const response = await fetch('/feedback', {
        method: 'POST', 
        headers: {
            'Content-Type': 'application/json'
        }, 
        body: JSON.stringify({"translation_id": translation_id, "liked": liked})
    });

    if (!response.ok) {
        console.error("Error sending feedback");
    }
});