//! text and markdown are already the rendition, they only need chunking

use crate::db::NewChunk;

const TARGET_CHARS: usize = 1200;

pub fn chunk_markdown(markdown: &str) -> Vec<NewChunk> {
    let mut chunks: Vec<NewChunk> = Vec::new();
    let mut heading: Option<String> = None;
    let mut buf = String::new();
    let mut buf_heading: Option<String> = None;
    let mut line_no: usize = 0;
    let mut buf_start_line: usize = 1;
    let mut in_fence = false;

    let mut flush = |buf: &mut String, heading: &Option<String>, start: usize| {
        let content = buf.trim();
        if !content.is_empty() {
            chunks.push(NewChunk {
                heading: heading.clone(),
                locator: Some(format!("line {start}")),
                content: content.to_string(),
            });
        }
        buf.clear();
    };

    for line in markdown.lines() {
        line_no += 1;
        let trimmed = line.trim_start();

        if trimmed.starts_with("```") {
            in_fence = !in_fence;
        }

        if !in_fence && trimmed.starts_with('#') {
            flush(&mut buf, &buf_heading, buf_start_line);
            heading = Some(trimmed.trim_start_matches('#').trim().to_string());
            buf_heading = heading.clone();
            buf_start_line = line_no;
            buf.push_str(line);
            buf.push('\n');
            continue;
        }

        if buf.is_empty() {
            buf_start_line = line_no;
            buf_heading = heading.clone();
        }
        buf.push_str(line);
        buf.push('\n');

        if !in_fence && buf.len() >= TARGET_CHARS && trimmed.is_empty() {
            flush(&mut buf, &buf_heading, buf_start_line);
        }
    }
    flush(&mut buf, &buf_heading, buf_start_line);

    chunks
}

pub fn naive_summary(markdown: &str) -> Option<String> {
    let para = markdown
        .split("\n\n")
        .map(str::trim)
        .find(|p| !p.is_empty() && !p.starts_with('#'))?;
    let mut s: String = para.chars().take(400).collect();
    if para.chars().count() > 400 {
        s.push('…');
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_headings() {
        let md = "# One\n\nalpha\n\n# Two\n\nbeta\n";
        let chunks = chunk_markdown(md);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading.as_deref(), Some("One"));
        assert!(chunks[0].content.contains("alpha"));
        assert_eq!(chunks[1].heading.as_deref(), Some("Two"));
    }

    #[test]
    fn keeps_fenced_blocks_whole() {
        let body = "x\n".repeat(1000);
        let md = format!("```\n{body}```\n\ntail\n");
        let chunks = chunk_markdown(&md);
        assert!(chunks[0].content.contains("```"));
        assert_eq!(chunks[0].content.matches("```").count(), 2);
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(chunk_markdown("   \n\n").is_empty());
    }
}
