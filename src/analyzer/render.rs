//! PDF page rasterization via hayro (pure Rust, CPU only).
//!
//! Page images are what make PDFs work on endpoints that reject native
//! documents, and what captures figures, charts and non-selectable maths.
//! Rendering is slow and can stumble on odd files, so every page is isolated:
//! one bad page never kills the rest, and zero rendered pages just means the
//! caller falls back to the document or text path.

use std::panic::AssertUnwindSafe;

/// Roughly 144 DPI. Legible for transcription and figures without huge PNGs.
const RENDER_SCALE: f32 = 2.0;
/// A single noisy page image never grows past this; oversize pages are skipped.
const MAX_PNG_BYTES: usize = 8 * 1024 * 1024;

pub struct PagePng {
    /// Zero-based page index, so headers stay truthful when pages are skipped.
    pub index: usize,
    pub png: Vec<u8>,
}

/// How many pages the file has, or zero when hayro cannot parse it at all.
pub fn page_count(bytes: &[u8]) -> usize {
    match hayro::hayro_syntax::Pdf::new(bytes.to_vec()) {
        Ok(pdf) => pdf.pages().len(),
        Err(_) => 0,
    }
}

/// Render one slice of the document, in order.
///
/// Rendering a whole book at once would hold every page image in memory at the
/// same time, so the analyzer asks for a batch, sends it, and comes back for
/// the next. The file is reparsed per batch, which is cheap next to rendering.
/// Pages that fail are skipped, so the result can be shorter than asked for.
pub fn render_pdf_range(bytes: &[u8], start: usize, count: usize) -> Vec<PagePng> {
    let Ok(pdf) = hayro::hayro_syntax::Pdf::new(bytes.to_vec()) else {
        return Vec::new();
    };
    let cache = hayro::RenderCache::new();
    let interp = hayro::hayro_interpret::InterpreterSettings::default();
    let mut pages = Vec::new();
    for (index, page) in pdf.pages().iter().enumerate().skip(start).take(count) {
        let rendered = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let pix = hayro::render(
                page,
                &cache,
                &interp,
                &hayro::RenderSettings {
                    x_scale: RENDER_SCALE,
                    y_scale: RENDER_SCALE,
                    bg_color: hayro::vello_cpu::color::palette::css::WHITE,
                    ..Default::default()
                },
            );
            encode_png(pix.width() as u32, pix.height() as u32, pix.data_as_u8_slice())
        }));
        if let Ok(Ok(png)) = rendered {
            pages.push(PagePng { index, png });
        }
    }
    pages
}

fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || width > 8000 || height > 8000 {
        return Err("page has absurd dimensions".into());
    }
    if rgba.len() != width as usize * height as usize * 4 {
        return Err("pixmap size mismatch".into());
    }
    // The pixmap is premultiplied, but the background is opaque white, so
    // every alpha is 255 and the bytes are plain RGBA.
    let mut buf = Vec::new();
    let mut enc = png::Encoder::new(&mut buf, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .map_err(|e| e.to_string())?
        .write_image_data(rgba)
        .map_err(|e| e.to_string())?;
    if buf.len() > MAX_PNG_BYTES {
        return Err("page image too large".into());
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-page PDF with a red square and no fonts, with a correct xref
    /// table so the parser has no excuse.
    fn tiny_pdf() -> Vec<u8> {
        let content = b"1 0 0 rg 0 0 100 100 re f\n";
        let objects: Vec<Vec<u8>> = vec![
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".to_vec(),
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n".to_vec(),
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
               /Contents 4 0 R >>\nendobj\n"
                .to_vec(),
            [
                format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).into_bytes(),
                content.to_vec(),
                b"\nendstream\nendobj\n".to_vec(),
            ]
            .concat(),
        ];
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for obj in &objects {
            offsets.push(pdf.len());
            pdf.extend_from_slice(obj);
        }
        let xref_at = pdf.len();
        pdf.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets {
            pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF",
                objects.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }

    #[test]
    fn renders_a_page_to_png() {
        let pdf = tiny_pdf();
        assert_eq!(page_count(&pdf), 1);
        let pages = render_pdf_range(&pdf, 0, 4);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].index, 0);
        // PNG magic, and big enough to hold a 400x400 image.
        assert_eq!(&pages[0].png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(pages[0].png.len() > 1000);
    }

    #[test]
    fn a_range_past_the_end_is_simply_empty() {
        let pdf = tiny_pdf();
        assert!(render_pdf_range(&pdf, 4, 4).is_empty());
    }

    #[test]
    fn garbage_input_yields_nothing_without_panicking() {
        assert_eq!(page_count(b"this is not a pdf at all"), 0);
        assert!(render_pdf_range(b"this is not a pdf at all", 0, 4).is_empty());
        assert_eq!(page_count(b""), 0);
        assert!(render_pdf_range(b"", 0, 4).is_empty());
    }
}
