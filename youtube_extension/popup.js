const chatBox =
document.getElementById("chat-box");

let currentVideoURL = "";

/* AUTO DETECT YOUTUBE URL */

async function getCurrentVideoURL() {

    try {

        const tabs =
        await chrome.tabs.query({

            active: true,
            currentWindow: true
        });

        const currentTab =
        tabs[0];

        if(
            currentTab.url.includes(
                "youtube.com/watch"
            ) ||

            currentTab.url.includes(
                "youtube.com/shorts"
            ) ||

            currentTab.url.includes(
                "youtu.be"
            )
        ) {

            currentVideoURL =
            currentTab.url;

            loadVideo(
                currentVideoURL
            );
        }

        else {

            chatBox.innerHTML = `

            <div class="system">

                ⚠️ Open a YouTube video first

            </div>
            `;
        }

    } catch(error) {

        console.log(error);

        chatBox.innerHTML = `

        <div class="system">

            ❌ Failed to detect URL

        </div>
        `;
    }
}

/* LOAD VIDEO */

async function loadVideo(videoUrl) {

    if(!videoUrl) return;

    let videoId = "";

    try {

        const url =
        new URL(videoUrl);

        /* Normal URL */

        if(
            url.searchParams.get("v")
        ) {

            videoId =
            url.searchParams.get("v");
        }

        /* Shorts */

        else if(
            url.pathname.includes(
                "/shorts/"
            )
        ) {

            videoId =
            url.pathname
            .split("/shorts/")[1];
        }

        /* youtu.be */

        else if(
            url.hostname.includes(
                "youtu.be"
            )
        ) {

            videoId =
            url.pathname.slice(1);
        }

    } catch(error) {

        console.log(
            "Invalid URL"
        );

        return;
    }

    /* Video Card */

    chatBox.innerHTML = `

    <div class="video-card">

        <img
            class="thumbnail"

            src="
https://img.youtube.com/vi/${videoId}/hqdefault.jpg
            "

            onerror="
this.src='https://i.imgur.com/8Km9tLL.png'
            "
        >

        <div class="video-info">

            <h3>
                Video Loaded 🚀
            </h3>

            <p>
                AI Ready to Chat
            </p>

        </div>

    </div>

    <div class="system">

        ⏳ Loading transcript...

    </div>
    `;

    chatBox.scrollTop =
    chatBox.scrollHeight;

    try {

        const formData =
        new FormData();

        formData.append(
            "video_url",
            videoUrl
        );

        const response =
        await fetch(

            "http://127.0.0.1:5001/load_video",

            {
                method: "POST",
                body: formData
            }
        );

        const data =
        await response.json();

        /* Remove loading */

        document
        .querySelector(".system")
        .remove();

        chatBox.innerHTML += `

        <div class="system">

            ${data.status}

        </div>
        `;

    } catch(error) {

        chatBox.innerHTML += `

        <div class="system">

            ❌ Failed to load video

        </div>
        `;
    }

    chatBox.scrollTop =
    chatBox.scrollHeight;
}

/* ASK QUESTION */

document
.getElementById("askBtn")

.addEventListener("click", async () => {

    const question =
    document.getElementById(
        "question"
    ).value.trim();

    if(!question) return;

    /* User Message */

    chatBox.innerHTML += `

    <div class="user">

        ${question}

    </div>
    `;

    document.getElementById(
        "question"
    ).value = "";

    /* Thinking */

    const loadingDiv =
    document.createElement("div");

    loadingDiv.className =
    "bot";

    loadingDiv.innerHTML =
    "✨ Thinking...";

    chatBox.appendChild(
        loadingDiv
    );

    chatBox.scrollTop =
    chatBox.scrollHeight;

    try {

        const formData =
        new FormData();

        formData.append(
            "message",
            question
        );

        const response =
        await fetch(

            "http://127.0.0.1:5001/predict",

            {
                method: "POST",
                body: formData
            }
        );

        const data =
        await response.json();

        loadingDiv.innerHTML =

        data.response
        .replace(/\n/g, "<br>");

    } catch(error) {

        loadingDiv.innerHTML =

        "❌ Error generating response";
    }

    chatBox.scrollTop =
    chatBox.scrollHeight;
});

/* ENTER KEY */

document
.getElementById("question")

.addEventListener(
    "keydown",

    (e) => {

        if(
            e.key === "Enter" &&
            !e.shiftKey
        ) {

            e.preventDefault();

            document
            .getElementById(
                "askBtn"
            )
            .click();
        }
    }
);

/* CLEAR CHAT */

document
.getElementById("clearBtn")

.addEventListener("click", () => {

    chatBox.innerHTML = `

    <div class="bot">

        👋 Chat cleared

    </div>
    `;
});

/* AUTO START */

window.onload = () => {

    getCurrentVideoURL();
};