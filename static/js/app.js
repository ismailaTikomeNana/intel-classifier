const imageInput = document.getElementById("imageInput");
const fileBtn = document.getElementById("fileBtn");
const fileName = document.getElementById("fileName");

const submitBtn = document.getElementById("submitBtn");

const resultCard = document.getElementById("resultCard");
const predClass = document.getElementById("predClass");
const predDesc = document.getElementById("predDesc");
const confValue = document.getElementById("confValue");
const confBar = document.getElementById("confBar");
const allProbs = document.getElementById("allProbs");

fileBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  fileName.textContent = imageInput.files[0]?.name || "No file selected";
});

submitBtn.addEventListener("click", async () => {
  if (!imageInput.files[0]) return;

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);
  formData.append("model", document.getElementById("modelSelect").value);

  const res = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  if (data.error) return alert(data.error);

  resultCard.style.display = "block";

  predClass.textContent = data.predicted_class;
  predDesc.textContent = data.description || "";

  confValue.textContent = data.confidence.toFixed(1) + "%";
  confBar.style.width = data.confidence + "%";

  allProbs.innerHTML = "";

  Object.entries(data.all_probs)
    .sort((a,b) => b[1]-a[1])
    .forEach(([key,val]) => {

      const row = document.createElement("div");
      row.className = "prob-row";

      row.innerHTML = `
        <div>${key}</div>
        <div class="small-bar">
          <div class="small-fill" style="width:${val}%"></div>
        </div>
        <div>${val.toFixed(1)}%</div>
      `;

      allProbs.appendChild(row);
    });
});