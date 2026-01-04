# LoreWeaver Frontend

This is the Streamlit-based frontend for LoreWeaver.

## Setup

1.  Ensure you have the project dependencies installed. You also need `streamlit` and `pyvis`.
    ```bash
    pip install streamlit pyvis networkx
    ```

## Running the App

Run the following command from the project root directory:

```bash
streamlit run frontend/app.py
```

## Features

*   **Pipeline Control**: Configure and run the LoreWeaver pipeline stages.
*   **Graph Visualization**: Interactive 2D/3D visualization of the generated knowledge graphs.
*   **Entity Inspector**: View detailed information about specific nodes in the graph.
