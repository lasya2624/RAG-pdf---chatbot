import { NextResponse } from 'next/server';
import pdf from 'pdf-parse';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { localEmbeddings } from '@/lib/embeddings';

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    console.log(`Starting ingestion for file: ${file.name}`);
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const data = await pdf(buffer);
    const text = data.text?.trim();

    console.log(`Extracted ${text?.length || 0} characters from PDF.`);
    if (!text || text.length < 5) {
      console.error("PDF extraction returned empty or insufficient text.");
      return NextResponse.json({ 
        error: "Could not extract text from PDF.", 
        details: "The document might be an image/scanned PDF, or it may be empty. Please try a text-based PDF." 
      }, { status: 400 });
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const docs = await splitter.createDocuments([text]);
    console.log(`Split document into ${docs.length} chunks.`);

    console.log("Generating embeddings locally using Transformers.js...");

    const chromaUrl = process.env.CHROMA_URL || "http://localhost:8000";
    
    const { ChromaClient } = await import('chromadb');
    const { chromaEmbeddingFunction } = await import('@/lib/embeddings');
    
    // ✅ FIX: Use ChromaClient to delete collection properly
    const client = new ChromaClient({ path: chromaUrl });
    try {
      await client.deleteCollection({ name: "documents" });
      console.log("Deleted existing collection 'documents'");
    } catch (e: any) {
      console.log("Collection check/delete: " + (e.message || "No existing collection found"));
    }

    // ✅ Create new collection natively with explicit embedding function to fix warning
    const collection = await client.createCollection({
      name: "documents",
      embeddingFunction: chromaEmbeddingFunction
    });

    const textsToEmbed = docs.map(doc => doc.pageContent);
    const docIds = docs.map((_, i) => `chunk_${i}`);

    await collection.add({
      ids: docIds,
      documents: textsToEmbed,
    });

    console.log("Ingestion complete.");
    return NextResponse.json({
      success: true,
      message: `Successfully indexed ${docs.length} chunks from ${file.name}`
    });

  } catch (error: any) {
    console.error("Ingestion error:", error);

    // Handle specific ChromaDB errors
    if (error.message?.includes("collection") || error.message?.includes("already exists")) {
      return NextResponse.json({
        error: "Collection already exists. Please restart ChromaDB or use a different collection name.",
        details: error.message
      }, { status: 500 });
    }

    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}