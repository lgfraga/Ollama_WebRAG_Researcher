"""
Autor: Luis González Fraga

Asunciones:
- Requiere Python >= 3.13
"""
import os
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass, fields
from dataclasses import dataclass
from typing import Any, Optional



DEFAULT_REPORT_STRUCTURE = """
<!-- ============================================================= -->
<!-- PLANTILLA DE RESUMEN O NOTAS EXTENSAS DE ESTUDIOS (RAG + FUENTES DOCUMENTALES) -->
<!-- Lista para publicación web / blog                             -->
<!-- Combina información de RAG y fuentes externas verificables     -->
<!-- ============================================================= -->

# Título del artículo
<!-- Instrucción: Título claro y atractivo (máx. 90 caracteres). Ejemplo: "Nuevas fronteras de la energía cuántica: lo que revelan los estudios recientes". -->

# Subtítulo o bajada
<!-- Instrucción: Frase que amplíe el tema o destaque el valor del resumen. -->

# Autor / Medio / Fecha
<!-- Instrucción: Nombre del autor, institución o medio, y fecha de publicación. -->

---

## Resumen ejecutivo
- **Propósito:** sintetizar los hallazgos más recientes sobre el tema en cuestión.  
- **Origen de la información:** combinación de documentos RAG, artículos científicos, prensa libre y portales web especializados.  
- **Relevancia:** explicar por qué el estudio o campo es significativo.  
<!-- Instrucción: Redactar entre 120 y 180 palabras; tono riguroso y comprensible. -->

---

## Introducción
- **Tema central:** descripción general del área o fenómeno estudiado.  
- **Contexto:** antecedentes del campo o evolución de la investigación.  
- **Justificación:** por qué se necesita revisar y resumir esta información ahora.  
<!-- Instrucción: Redactar 2–3 párrafos que sitúen al lector en el tema. -->

---

## Fuentes y metodología de recopilación
- **Origen de los datos:** resultados del RAG (resúmenes, papers, informes) y fuentes externas verificables.  
- **Tipos de documentos consultados:** artículos científicos, informes técnicos, notas de prensa y artículos divulgativos.  
- **Criterios de selección:** actualidad, fiabilidad, acceso libre, impacto y relevancia temática.  
- **Método de análisis:** lectura crítica, extracción de conceptos clave, clasificación temática y síntesis.  
<!-- Instrucción: Explicar brevemente cómo se combinan las fuentes internas y externas. -->

---

## Desarrollo y análisis del contenido

### 1. Antecedentes y evolución del tema
- Principales hitos o descubrimientos en el campo.  
- Cambios teóricos, metodológicos o tecnológicos recientes.  
- Contexto histórico o científico que dio origen a los estudios.  
<!-- Instrucción: Integrar información histórica o fundacional basada en el RAG o en fuentes académicas. -->

### 2. Principales hallazgos documentales
- Síntesis de resultados de estudios o informes más relevantes.  
- Datos cuantitativos y cualitativos clave.  
- Coincidencias o discrepancias entre autores o instituciones.  
<!-- Instrucción: Presentar los hallazgos de forma ordenada y verificable. -->

### 3. Análisis comparativo
- Diferencias metodológicas o conceptuales entre las fuentes.  
- Contraste entre resultados del RAG y artículos externos.  
- Evaluación de la coherencia y solidez de las conclusiones.  
<!-- Instrucción: Analizar las fuentes críticamente y resaltar convergencias o vacíos. -->

### 4. Perspectivas y tendencias actuales
- Nuevas líneas de investigación emergentes.  
- Aplicaciones prácticas, tecnológicas o sociales.  
- Problemas aún no resueltos o desafíos futuros.  
<!-- Instrucción: Destacar hacia dónde se dirige el campo según los datos analizados. -->

---

## Discusión
- **Interpretación global:** cómo se integran los hallazgos del RAG con los de las fuentes documentales.  
- **Impacto académico y social:** por qué estos estudios son relevantes.  
- **Limitaciones:** lagunas de información, sesgos o restricciones de acceso.  
<!-- Instrucción: Ofrecer una lectura crítica y global del panorama actual. -->

---

## Conclusiones
- **Síntesis final:** principales aportes del resumen y su relevancia interdisciplinar.  
- **Reflexión:** cómo los hallazgos influyen en el conocimiento general del tema.  
- **Futuras líneas de trabajo:** sugerencias para nuevas investigaciones o aplicaciones.  
<!-- Instrucción: Redactar conclusiones claras y concisas en 2–3 párrafos. -->

---

## Fuentes y referencias
- [1] Autor, A. (Año). *Título del estudio*. Revista / Repositorio, volumen(número), DOI o URL.  
- [2] Organización o institución (Año). *Informe o publicación oficial*. URL.  
- [3] Medio de comunicación (fecha). *Título del artículo*. URL.  
- [4] Documento RAG interno (Año / ID). *Título o descripción resumida*.  
<!-- Instrucción: Incluir fuentes tanto externas como internas (RAG), priorizando acceso libre y trazabilidad. -->

---

## Recursos visuales y multimedia
- **Gráfico o tabla:** resumen de hallazgos clave.  
- **Mapa conceptual:** estructura temática del estudio.  
- **Imagen destacada:** fotografía o infografía con licencia abierta.  
<!-- Instrucción: Indicar origen, autoría y licencia (Creative Commons o dominio público). -->

---

## Metadatos para publicación web
- **Slug URL:** ejemplo: /resumen-estudios-RAG-inteligencia-artificial-2025  
- **Categorías / etiquetas:** ciencia, tecnología, educación, medio ambiente, innovación.  
- **Imagen destacada:** nombre de archivo o URL.  
- **Descripción SEO (150–160 caracteres):** resumen optimizado para motores de búsqueda.  
- **Extracto breve (1 párrafo):** texto para portada o redes sociales.  
<!-- Instrucción: Completar todos los campos antes de publicar en CMS. -->

---

# Puntos clave (Key Takeaways)
- <!-- Instrucción: Escribir entre 4 y 6 viñetas con los principales hallazgos o conclusiones del resumen. -->

---

# Nota operativa para el agente
<!--
1) Combinar los datos internos del RAG con fuentes externas abiertas y verificables.
2) Priorizar información contrastada y reciente; citar cada fuente claramente.
3) Redactar en tono analítico y divulgativo: claridad, rigor y cohesión temática.
4) Evitar repeticiones o redundancias; sintetizar ideas complejas en lenguaje accesible.
5) Mantener formato Markdown/HTML y agregar metadatos SEO para publicación directa.
6) Si se detectan inconsistencias entre fuentes, explicarlas sin omitir la referencia.
-->

"""

@dataclass(kw_only=True)
class Configuration:
    """Los campos configurables para el chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    max_search_queries: int = 5
    enable_web_search: bool = False

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Crear una instancia de Configuración a partir de un RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})