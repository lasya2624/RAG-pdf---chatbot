const { GoogleGenerativeAI } = require("@google/generative-ai");
require("dotenv").config({ path: ".env.local" });

async function listModels() {
  const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.error("No API key found in .env.local");
    return;
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  
  try {
    console.log("Fetching available models for your API key...");
    const models = await genAI.listModels();
    
    console.log("\nAvailable Models:");
    console.table(models.models.map(m => ({
      name: m.name.replace('models/', ''),
      description: m.description,
      supportedMethods: m.supportedGenerationMethods.join(', ')
    })));
  } catch (error) {
    console.error("Error fetching models:", error.message);
  }
}

listModels();
