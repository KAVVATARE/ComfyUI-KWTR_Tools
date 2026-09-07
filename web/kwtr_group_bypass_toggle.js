import { app } from "../../scripts/app.js";

// LiteGraph mode constants (see rgthree-comfy's fast_groups_muter.ts for the
// same convention): ALWAYS=0 runs the node normally, 4 is the mode ComfyUI
// uses for "bypass" (pass inputs straight through, skip execution).
const MODE_ENABLED = 0;
const MODE_BYPASS = 4;

function getGroupTitles(graph) {
    const groups = graph?._groups || graph?.groups || [];
    return groups.map((g) => g.title).filter(Boolean);
}

function applyBypassToGroup(graph, groupTitle, enabled) {
    if (!graph || !groupTitle) return;
    const groups = graph._groups || graph.groups || [];
    for (const group of groups) {
        if (group.title !== groupTitle) continue;
        group.recomputeInsideNodes?.();
        const children = group._children ? Array.from(group._children) : (group._nodes || []);
        for (const node of children) {
            if (node instanceof LGraphNode) {
                node.mode = enabled ? MODE_ENABLED : MODE_BYPASS;
            }
        }
    }
}

app.registerExtension({
    name: "kwtr.GroupBypassToggle2",
    registerCustomNodes() {
        // Same on_false/on_true/switch layout as KJNodes' "Lazy Switch KJ",
        // but instead of routing a value it bypasses one of two named groups.
        class KWTRGroupBypassToggle2 extends LGraphNode {
            static title = "KWTR_GroupBypassToggle";
            static category = "utils";

            constructor(title) {
                super(title || KWTRGroupBypassToggle2.title);
                this.isVirtualNode = true;
                this.serialize_widgets = true;

                this.onFalseWidget = this.addWidget(
                    "combo",
                    "on_false",
                    "Audio",
                    () => this.applyState(this.switchWidget.value),
                    { values: [] },
                );
                this.onTrueWidget = this.addWidget(
                    "combo",
                    "on_true",
                    "Audio Refiner",
                    () => this.applyState(this.switchWidget.value),
                    { values: [] },
                );
                this.switchWidget = this.addWidget(
                    "toggle",
                    "switch",
                    true,
                    (value) => this.applyState(value),
                    { on: "true", off: "false" },
                );

                this.size = this.computeSize();
            }

            // Keep the two combo dropdowns showing the workflow's current
            // group titles (groups can be added/renamed/removed at any time).
            onDrawForeground(ctx) {
                const graph = this.graph || app.graph;
                const titles = getGroupTitles(graph);
                const list = titles.length ? titles : [""];
                this.onFalseWidget.options.values = list;
                this.onTrueWidget.options.values = list;
                return LGraphNode.prototype.onDrawForeground?.apply(this, arguments);
            }

            // switch=true -> on_true group runs, on_false group is bypassed.
            // switch=false -> the reverse. Always mutually exclusive.
            applyState(value) {
                const graph = this.graph || app.graph;
                applyBypassToGroup(graph, this.onTrueWidget.value, value);
                applyBypassToGroup(graph, this.onFalseWidget.value, !value);
                graph?.setDirtyCanvas(true, false);
            }

            // Fresh node dropped on the canvas: sync the two groups to the
            // widgets' default values right away.
            onAdded() {
                this.applyState(this.switchWidget.value);
            }

            // Workflow loaded from disk: widget values are restored just
            // before this fires, so re-apply with the persisted values.
            onConfigure() {
                this.applyState(this.switchWidget.value);
            }
        }

        LiteGraph.registerNodeType("KWTRGroupBypassToggle2", KWTRGroupBypassToggle2);
    },
});
