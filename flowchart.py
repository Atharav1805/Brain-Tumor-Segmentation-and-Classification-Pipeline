from graphviz import Digraph

dot = Digraph("UNet", format="svg")
dot.attr(rankdir="LR", size="12,6")

# ----- Nodes -----
dot.node("I", "Input\n256×256×1")

dot.node("E1", "Encoder Block 1\n(Conv→BN→ReLU)×2\n32 filters")
dot.node("E2", "Encoder Block 2\n64 filters")
dot.node("E3", "Encoder Block 3\n128 filters")
dot.node("E4", "Encoder Block 4\n256 filters")

dot.node("B", "Bottleneck\n512 filters")

dot.node("D1", "Decoder Block 1\nUpSample + Skip + Conv\n256 filters")
dot.node("D2", "Decoder Block 2\n128 filters")
dot.node("D3", "Decoder Block 3\n64 filters")
dot.node("D4", "Decoder Block 4\n32 filters")

dot.node("O", "Output Layer\n1×1 Conv + Sigmoid")

# ----- Main Flow (fixed format) -----
dot.edges([
    ("I","E1"),
    ("E1","E2"),
    ("E2","E3"),
    ("E3","E4"),
    ("E4","B"),
    ("B","D1"),
    ("D1","D2"),
    ("D2","D3"),
    ("D3","D4"),
    ("D4","O"),
])

# ----- Skip Connections -----
dot.edge("E1", "D4", style="dashed", label="skip")
dot.edge("E2", "D3", style="dashed", label="skip")
dot.edge("E3", "D2", style="dashed", label="skip")
dot.edge("E4", "D1", style="dashed", label="skip")

dot.render("unet_flowchart", cleanup=True)
