const checkbox = document.getElementById("only-alerts");
if (checkbox) {
  const shownCount = document.getElementById("shown-count");
  const updateShownCount = () => {
    const visibleRows = Array.from(document.querySelectorAll("#results-table tbody tr[data-alert]"))
      .filter((row) => row.style.display !== "none");
    if (shownCount) {
      shownCount.textContent = String(visibleRows.length);
    }
  };
  checkbox.addEventListener("change", () => {
    document.querySelectorAll("#results-table tbody tr[data-alert]").forEach((row) => {
      const isAlert = row.getAttribute("data-alert") === "1";
      row.style.display = checkbox.checked && !isAlert ? "none" : "";
    });
    updateShownCount();
  });
  updateShownCount();
}
