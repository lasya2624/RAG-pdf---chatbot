const { GoogleGenerativeAI } = require("@google/generative-ai");
require("dotenv").config({ path: ".env.local" });

async function test() {
  const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);
  const model = genAI.getGenerativeModel({ model: "embedding-001" });
  try {
    const result = await model.embedContent("Hello world");
    console.log("Success! Embedding length:", result.embedding.values.length);
  } catch (error) {
    console.error("Error:", error.message);
  }
}

test();
