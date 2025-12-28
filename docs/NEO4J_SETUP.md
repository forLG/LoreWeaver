# Neo4j Setup Guide

## Quick Start (Windows)

### Option 1: Neo4j Desktop (Recommended)

1. **Download**
   - Go to https://neo4j.com/download/
   - Download "Neo4j Desktop"

2. **Install & Create Database**
   ```
   - Install Neo4j Desktop
   - Create a new project (e.g., "LoreWeaver")
   - Click "Add Database" → Select version 5.x
   - Set a password (remember it!)
   - Click "Start"
   ```

3. **Update Config**
   ```python
   # Edit config_neo4j.py
   NEO4J_PASSWORD = "your_actual_password"
   ```

4. **Run Import**
   ```bash
   python scripts/build_graph.py --mode neo4j
   ```

### Option 2: Docker

```powershell
# Pull and run Neo4j container
docker run -d `
    --name neo4j `
    -p 7474:7474 -p 7687:7687 `
    -e NEO4J_AUTH=neo4j/your_password `
    neo4j:5.23-community
```

### Option 3: Neo4j Community Server

1. Download: https://neo4j.com/deployment-center/
2. Extract to a folder (e.g., `C:\neo4j`)
3. Run `bin\neo4j.bat console`
4. Set password when prompted
5. Update `config_neo4j.py`

## Verify Installation

Visit http://localhost:7474 and log in with:
- Username: `neo4j`
- Password: (your password)

## Troubleshooting

| Error | Solution |
|-------|----------|
| Port 7687 in use | Change port in `neo4j.conf` |
| Connection refused | Check if Neo4j is running |
| Auth failed | Verify password in `config_neo4j.py` |
