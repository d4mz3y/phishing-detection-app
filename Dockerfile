FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown user:user /home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt gunicorn

COPY --chown=user . .
USER user

# Frozen public DistilBERT weights are git-ignored; fetch them at build time.
RUN python download_text_encoder.py

EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
