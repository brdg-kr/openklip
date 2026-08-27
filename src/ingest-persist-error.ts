/** A project exists, but one required post-ingest step failed. */
export class IngestPartialError extends Error {
  readonly slug: string;

  constructor(name: string, slug: string, message: string) {
    super(message);
    this.name = name;
    this.slug = slug;
  }
}

/** Thrown when ingest succeeds but copying the upload to a durable source path fails. */
export class IngestPersistError extends IngestPartialError {
  constructor(slug: string, cause: unknown) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(
      "IngestPersistError",
      slug,
      `Project "${slug}" was created but saving the original source failed (${detail}). The editor works; exports fall back to the 720p proxy until you copy the source into the project folder or re-ingest.`
    );
  }
}

export function isIngestPartialError(
  error: unknown
): error is IngestPartialError {
  return error instanceof IngestPartialError;
}

export function isIngestPersistError(
  error: unknown
): error is IngestPersistError {
  return error instanceof IngestPersistError;
}
