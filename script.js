class WellLabelingTool {
    constructor() {
        this.canvas = document.getElementById("imageCanvas");
        this.ctx = this.canvas.getContext("2d");
        this.wells = [];
        this.currentImage = null;
        this.pendingClick = null;

        this.initEvents();
    }

    initEvents() {
        document.getElementById("imageUpload").addEventListener("change", e => this.loadImage(e));
        this.canvas.addEventListener("click", e => this.onCanvasClick(e));
        document.getElementById("wellForm").addEventListener("submit", e => this.addWell(e));
        document.getElementById("cancelBtn").addEventListener("click", () => this.closeModal());
        document.getElementById("exportBtn").addEventListener("click", () => this.exportCSV());
        document.getElementById("labelModal").addEventListener("click", e => {
            if (e.target === document.getElementById("labelModal")) this.closeModal();
        });
    }

    loadImage(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = e => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.canvas.width = img.width;
                this.canvas.height = img.height;
                this.wells = [];
                this.updateWellCount();
                document.getElementById("exportBtn").disabled = true;
                this.redraw();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    onCanvasClick(event) {
        if (!this.currentImage) return;
        const rect = this.canvas.getBoundingClientRect();
        this.pendingClick = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        this.showModal();
    }

    addWell(event) {
        event.preventDefault();
        const prefix = document.getElementById("wellPrefix").value;
        const number = document.getElementById("wellNumber").value;
        if (!prefix || !number || !this.pendingClick) return;

        const wellId = `${prefix}-${number}`;
        if (this.wells.some(w => w.well_id === wellId)) {
            alert("A well with this ID already exists.");
            return;
        }

        this.wells.push({
            well_id: wellId,
            x: Math.round(this.pendingClick.x),
            y: Math.round(this.pendingClick.y)
        });

        this.redraw();
        this.updateWellCount();
        document.getElementById("exportBtn").disabled = false;
        this.closeModal();
    }

    showModal() {
        document.getElementById("labelModal").style.display = "block";
        document.getElementById("wellPrefix").value = "GW";
        document.getElementById("wellNumber").value = "";
        document.getElementById("wellPrefix").focus();
    }

    closeModal() {
        document.getElementById("labelModal").style.display = "none";
        this.pendingClick = null;
    }

    redraw() {
        if (!this.currentImage) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.currentImage, 0, 0);
        this.wells.forEach(w => this.drawWell(w.x, w.y, w.well_id));
    }

    drawWell(x, y, label) {
        this.ctx.fillStyle = "red";
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, Math.PI * 2);
        this.ctx.fill();

        this.ctx.font = "14px Arial";
        this.ctx.lineWidth = 3;
        this.ctx.strokeStyle = "white";
        this.ctx.strokeText(label, x + 10, y - 5);
        this.ctx.fillStyle = "black";
        this.ctx.fillText(label, x + 10, y - 5);
    }

    updateWellCount() {
        document.getElementById("wellCount").textContent = this.wells.length;
    }

    exportCSV() {
        if (this.wells.length === 0) {
            alert("No wells to export.");
            return;
        }
        let csv = "well_id,x,y\n";
        this.wells.forEach(w => csv += `${w.well_id},${w.x},${w.y}\n`);

        const blob = new Blob([csv], { type: "text/csv" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "wells.csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
}

document.addEventListener("DOMContentLoaded", () => new WellLabelingTool());
