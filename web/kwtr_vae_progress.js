import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "VAEDecodeTiledProgress";
const BAR_COLOR = "#22c55e";
const BAR_HEIGHT = 6;

app.registerExtension({
    name: "kwtr.VAEDecodeTiledProgress",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed) return;

            const p = this._kwtrProgress ?? 0;
            if (p <= 0) return;

            const w = this.size[0];
            ctx.save();
            // faint track
            ctx.fillStyle = "rgba(34,197,94,0.15)";
            ctx.fillRect(0, 0, w, BAR_HEIGHT);
            // filled portion
            ctx.fillStyle = BAR_COLOR;
            ctx.fillRect(0, 0, w * Math.min(Math.max(p, 0), 1), BAR_HEIGHT);
            ctx.restore();
        };
    },

    setup() {
        const setProgress = (nodeId, value) => {
            if (nodeId == null) return;
            const node = app.graph.getNodeById(Number(nodeId));
            if (!node || node.type !== TARGET_NODE) return;
            node._kwtrProgress = value;
            app.graph.setDirtyCanvas(true, false);
        };

        // Standard tile progress from decode_tiled: { value, max, node }
        api.addEventListener("progress", ({ detail }) => {
            if (!detail || detail.max == null) return;
            const nodeId = detail.node;
            const node = app.graph.getNodeById(Number(nodeId));
            if (!node || node.type !== TARGET_NODE) return;
            const ratio = detail.max > 0 ? detail.value / detail.max : 0;
            setProgress(nodeId, ratio);
        });

        // Clear bar when this node starts (executing == this node id) is not
        // reliable for reset, so clear on executed and on execution end.
        api.addEventListener("executed", ({ detail }) => {
            setProgress(detail?.node, 0);
        });

        api.addEventListener("execution_success", () => {
            for (const node of app.graph._nodes) {
                if (node.type === TARGET_NODE) {
                    node._kwtrProgress = 0;
                }
            }
            app.graph.setDirtyCanvas(true, false);
        });

        // Also clear when a fresh prompt starts.
        api.addEventListener("execution_start", () => {
            for (const node of app.graph._nodes) {
                if (node.type === TARGET_NODE) {
                    node._kwtrProgress = 0;
                }
            }
            app.graph.setDirtyCanvas(true, false);
        });
    },
});
