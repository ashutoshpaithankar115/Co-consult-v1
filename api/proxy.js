export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
  if (!ANTHROPIC_API_KEY) return res.status(500).json({ error: 'NO_API_KEY' });

  try {
    const body = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
    let parsed;
    try { parsed = JSON.parse(body); } catch(e) { return res.status(400).json({ error: 'BAD_JSON', body: body.slice(0,200) }); }

    parsed.stream = false;
    parsed.model = 'claude-opus-4-5';

    const anthropicRes = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(parsed),
    });

    const responseText = await anthropicRes.text();

    // Always return 200 with full debug info
    return res.status(200).json({
      anthropic_status: anthropicRes.status,
      response: responseText.slice(0, 2000),
      model_used: parsed.model,
      key_prefix: ANTHROPIC_API_KEY.slice(0, 20),
    });

  } catch (err) {
    return res.status(200).json({ error: err.message });
  }
}
