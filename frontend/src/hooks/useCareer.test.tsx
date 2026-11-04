import { useCallback } from 'react'
import { act, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { useCareerAction, useCareerRead } from './useCareer'
const active=(value:string)=>value==='pending'
afterEach(()=>vi.useRealTimers())
it('locks duplicate submissions synchronously',async()=>{
 let resolve!:(value:string)=>void
 const task=vi.fn(()=>new Promise<string>(r=>{resolve=r})),accept=vi.fn(),{result}=renderHook(()=>useCareerAction())
 let pending:Promise<void>
 act(()=>{pending=result.current.run(task,accept);void result.current.run(task,accept)})
 expect(task).toHaveBeenCalledTimes(1)
 await act(async()=>{resolve('saved');await pending})
 expect(accept).toHaveBeenCalledExactlyOnceWith('saved')
})
it('aborts on unmount and refuses late mutation responses',async()=>{
 let resolve!:(value:string)=>void;let signal!:AbortSignal
 const task=(s:AbortSignal)=>{signal=s;return new Promise<string>(r=>{resolve=r})},accept=vi.fn(),view=renderHook(()=>useCareerAction())
 let pending:Promise<void>;act(()=>{pending=view.result.current.run(task,accept)})
 view.unmount();expect(signal.aborted).toBe(true)
 await act(async()=>{resolve('stale');await pending})
 expect(accept).not.toHaveBeenCalled()
})
it('polls GET state then stops at a terminal outcome',async()=>{
 vi.useFakeTimers();const load=vi.fn().mockResolvedValueOnce('pending').mockResolvedValue('complete'),view=renderHook(()=>useCareerRead(load,active))
 await act(async()=>{})
 expect(load).toHaveBeenCalledTimes(1)
 await act(async()=>{await vi.advanceTimersByTimeAsync(3000)})
 expect(view.result.current.value).toBe('complete')
 await act(async()=>{await vi.advanceTimersByTimeAsync(60000)})
 expect(load).toHaveBeenCalledTimes(2)
})
it('pauses after forty status reads and never retries an error automatically',async()=>{
 vi.useFakeTimers();const load=vi.fn().mockResolvedValue('pending'),view=renderHook(()=>useCareerRead(load,active))
 await act(async()=>{})
 await act(async()=>{await vi.advanceTimersByTimeAsync(120000)})
 expect(load).toHaveBeenCalledTimes(40);expect(view.result.current.paused).toBe(true)
 load.mockRejectedValueOnce(Object.assign(new Error('expired'),{status:401}))
 await act(async()=>{view.result.current.refresh()})
 expect(view.result.current.value).toBeNull();expect(view.result.current.error).toContain('session expired')
 await act(async()=>{await vi.advanceTimersByTimeAsync(60000)})
 expect(load).toHaveBeenCalledTimes(41)
})
it('aborts prior scope reads and ignores responses from old navigation',async()=>{
 let oldResolve!:(value:string)=>void;let oldSignal!:AbortSignal
 const load=(id:string,s:AbortSignal)=>id==='old'?new Promise<string>(r=>{oldResolve=r;oldSignal=s}):Promise.resolve('new data')
 const view=renderHook(({id})=>useCareerRead(useCallback((s:AbortSignal)=>load(id,s),[id])),{initialProps:{id:'old'}})
 await act(async()=>{view.rerender({id:'new'})})
 expect(oldSignal.aborted).toBe(true);expect(view.result.current.value).toBe('new data')
 await act(async()=>{oldResolve('old data')})
 expect(view.result.current.value).toBe('new data')
})
