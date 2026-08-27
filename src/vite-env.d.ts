/// <reference types="vite/client" />

declare module '*?worker' {
  const workerConstr: {
    new (): Worker
  }
  export default workerConstr
}
