"""Weekly product-update marketing email — migrated from the Node backend cron.

Node's cron-jobs/marketing/product-updates.ts previously called OpenAI gpt-4
directly to turn the week's git commits into a customer-facing HTML email. That
model logic now lives here and runs through OpenRouter.

Node still owns the rest: reading the git log, wrapping the returned HTML in the
DentNode email chrome (header/body/footer partials), and sending to every lab.

Both prompts are copied VERBATIM from the Node implementation; only the provider
changed. Note this is the one AI call in the platform with NO owning lab — see
PLATFORM_LAB_ID in server.py for how it is attributed in the usage ledger.
"""

import json
from typing import Any, Dict, List

from .config import settings
from .openrouter import ChatResult, chat_completion

# Copied verbatim from product-updates.ts (the system message).
SYSTEM_PROMPT = (
    "You are an expert marketing copywriter specializing in SaaS product updates "
    "for healthcare/dental industry. You write warm, friendly emails that "
    "translate technical updates into customer benefits."
)


def build_user_prompt(commits: List[Dict[str, Any]]) -> str:
    """Verbatim port of the prompt template in generateAIMessage()."""
    return f"""You are a marketing copywriter for DentNode, a dental lab management software platform.

DentNode helps dental labs:
- Manage cases efficiently with FDI, Palmer, and Universal tooth notation support
- Track workflows with automated workflow automation
- Handle billing and accounting
- Enable clinicians to submit cases through DN Clinic portal
- Manage staff and technicians through DN Workbench app
- Support international operations with multiple currencies and languages
- Provide mobile and desktop access for dental lab operations

Based on the following technical commit logs from our development team, write a warm, friendly, and professional weekly product update email in HTML format.

The email should:
1. Be addressed to dental lab owners (non-technical audience)
2. Explain what improvements we've made in simple terms
3. Highlight how these changes will benefit their dental lab operations
4. Use a conversational, friendly tone (like you're explaining to a friend)
5. Include a clear call-to-action to try the new features
6. Be structured with:
   - A warm greeting
   - Brief introduction about weekly improvements
   - 2-3 main feature highlights with benefits (use headings)
   - Closing with support information
7. Use professional HTML email formatting with:
   - Proper heading tags (h2, h3)
   - Paragraphs with good spacing
   - Bullet points where appropriate
   - Professional color scheme (blues, whites, subtle accents)
   - Mobile-responsive inline styles

Here are the commits from this week:
{json.dumps(commits, indent=2, ensure_ascii=False)}

IMPORTANT:
- Do NOT mention technical terms like "API", "database", "schema", "controllers", "hooks", "components"
- Focus on USER-FACING benefits like "faster invoice generation", "easier case tracking", "improved subscription management"
- Make it feel like a personal update from the DentNode team
- Keep it under 500 words
- Return ONLY the HTML email content (no markdown, no explanations)"""


async def generate_product_update_email(
    *, commits: List[Dict[str, Any]]
) -> ChatResult:
    """Generate the inner HTML for the weekly product-update email.

    Node wraps the returned HTML in the email chrome. Raises OpenRouterError on
    model failure so the endpoint can meter status="error"; the Node cron already
    treats a generation failure as fatal for that run.
    """
    return await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(commits)},
        ],
        model=settings.model,
        temperature=0.7,
        max_tokens=2000,
    )
