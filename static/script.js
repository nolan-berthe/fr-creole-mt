const EXAMPLES = [
  "Le soleil brille maintenant",
  "Alors, Je suis prêt",
  "Rien de nouveau s'il te plaît",
  "Je ne comprends pas maintenant",
  "C'est moi !",
  "Ce n'est pas bon !",
  "Elle est gentille !",
  "Est-ce que je vais travailler",
];

const input = document.getElementById("input-text");
const button = document.getElementById("translate-btn");
const status = document.getElementById("status");
const result = document.getElementById("result");
const chips = document.getElementById("chips");

function renderChips() {
  EXAMPLES.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      input.value = text;
      doTranslate();
    });
    chips.appendChild(chip);
  });
}

async function doTranslate() {
  const text = input.value.trim();
  if (!text) {
    status.textContent = "Écris une phrase d'abord.";
    status.className = "status error";
    return;
  }

  button.disabled = true;
  status.textContent = "Traduction en cours…";
  status.className = "status";

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      status.textContent = data.error || "Erreur.";
      status.className = "status error";
      result.textContent = "—";
      return;
    }

    result.textContent = data.translation || "—";
    status.textContent = "";
  } catch (err) {
    status.textContent = "Impossible de contacter le serveur.";
    status.className = "status error";
  } finally {
    button.disabled = false;
  }
}

button.addEventListener("click", doTranslate);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    doTranslate();
  }
});

renderChips();
