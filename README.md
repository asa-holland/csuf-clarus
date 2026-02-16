# Clarus

Clarus in association with CSU Fullerton's Masters of Software Engineering program.

## Quick Start

### Using Docker Compose

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd csuf-clarus
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**

   The application will be available at `http://localhost:5000`

### Stopping the Application

To stop the application, press `Ctrl+C` in the terminal where Docker Compose is running, or run:

```bash
docker-compose down
```

### Development Mode

The Docker Compose configuration is set up for development with:
- Live reloading enabled
- Debug mode active
- Volume mounts for code changes
- Development environment variables

### Requirements

- Docker
- Docker Compose
