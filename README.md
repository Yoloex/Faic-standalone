# Faic

```mermaid
flowchart TD
    FE[Frontend apps] -->|events| BME[bme-app]
    Team[Internal team] -.->|brand| BME
    BME -->|Routing| CS[Context-Service (FastAPI)]
    BME -->|Send stream| RS[Redis streams]
    RS --> RDB[Redis Context DB<br/>(Short-term + TTL)]
    RDB -.->|Read delta| CS
    CS -->|Store brand context| PG[PostgreSQL]

    %% Styling for clarity
    style FE fill:#a3d5f7,stroke:#333,stroke-width:1px
    style BME fill:#f9b6a3,stroke:#333,stroke-width:1px
    style CS fill:#c9b6f9,stroke:#333,stroke-width:1px
    style PG fill:#f9d6a3,stroke:#333,stroke-width:1px
    style RS fill:#ffe6cc,stroke:#333,stroke-width:1px
    style RDB fill:#ffe6cc,stroke:#333,stroke-width:1px

