import { pipeline } from '@xenova/transformers';
import { Embeddings } from "@langchain/core/embeddings";

class LocalHuggingFaceEmbeddings extends Embeddings {
  private pipelinePromise: Promise<any> | null = null;
  model = "Xenova/all-MiniLM-L6-v2";

  constructor() {
    super({});
  }

  private async getPipeline() {
    if (!this.pipelinePromise) {
      this.pipelinePromise = pipeline('feature-extraction', this.model);
    }
    return this.pipelinePromise;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const extractor = await this.getPipeline();
    const results = [];

    for (const text of texts) {
      const output = await extractor(text, { pooling: 'mean', normalize: true });
      results.push(Array.from(output.data) as number[]);
    }

    return results;
  }

  async embedQuery(text: string): Promise<number[]> {
    const embeddings = await this.embedDocuments([text]);
    return embeddings[0];
  }
}

export const localEmbeddings = new LocalHuggingFaceEmbeddings();

// Export a native ChromaDB embedding function wrapper to avoid warnings
export const chromaEmbeddingFunction = {
  generate: async (texts: string[]) => {
    return await localEmbeddings.embedDocuments(texts);
  }
};
